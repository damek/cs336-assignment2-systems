import torch, triton, os
import triton.language as tl
import math
from einops import rearrange, einsum

uid = getattr(os, "getuid", lambda: os.getpid())()
cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{uid}")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
os.environ.setdefault("USER", f"user{uid}")   # sidestep getpass.getuser()
os.environ.setdefault("HOME", "/tmp")
os.makedirs(cache_dir, exist_ok=True)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal : tl.constexpr
    ):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
    Q_ptr + batch_index * stride_qb,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
    O_ptr + batch_index * stride_ob,
    shape=(N_QUERIES, D),
    strides=(stride_oq, stride_od),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
    L_ptr + batch_index * stride_lb,
    shape=(N_QUERIES,),
    strides=(stride_lq,),
    offsets=(query_tile_index * Q_TILE_SIZE,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
    K_ptr + batch_index * stride_kb,
    shape=(N_KEYS, D),
    strides=(stride_kk, stride_kd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
    V_ptr + batch_index * stride_vb,
    shape=(N_KEYS, D),
    strides=(stride_vk, stride_vd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0),
    )

    m_i = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    idx_q_base = tl.arange(0, Q_TILE_SIZE)
    idx_k_base = tl.arange(0, K_TILE_SIZE)
    idx_q = idx_q_base + Q_TILE_SIZE*query_tile_index
    mask_scale = -1e6

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        S = tl.dot(Q_i,tl.trans(K_j))*scale
        if is_causal: 
            idx_k = idx_k_base + j*K_TILE_SIZE
            mask = mask_scale*(idx_k[None,:] > idx_q[:,None])
            S += mask
        m_i_new = tl.maximum(m_i, tl.max(S, axis=-1))
        tildeP = tl.exp(S - m_i_new[:, None])
        exp_scale = tl.exp(m_i - m_i_new)
        l_i = exp_scale*l_i + tl.sum(tildeP, axis=-1)
        O_i = O_i*exp_scale[:, None]
        O_i += tl.dot(tildeP.to(V_j.dtype), V_j)
        m_i = m_i_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    l_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, O_i, boundary_check=(0,1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))


# @torch.compile
# def flash_attention_backward(Q, K, V, L, O, dO,sqrt_d,is_causal):
#     D = (O * dO).sum(dim=-1, keepdim=True)
#     S = einsum(Q, K, "... i d, ... j d -> ... i j")/sqrt_d
#     if is_causal: 
#         i = torch.arange(S.shape[-2], device=S.device)
#         mask = i[None, :] > i[:, None]
#         S = S.masked_fill(mask, float("-inf"))
#     P = torch.exp(S - L.unsqueeze(-1))
#     dV = einsum(P, dO, "... i j, ... i d -> ... j d")
#     dP = einsum(dO, V, "... i d, ... j d -> ... i j")
#     dS = P * (dP - D)
#     dQ = einsum(dS, K, "... a b, ... b c-> ... a c")/sqrt_d
#     dK = einsum(dS, Q, "... a b, ... a d -> ... b d")/sqrt_d

#     return dQ, dK, dV, None

@torch.compile
def flash_attention_backward(Q, K, V, L, O, dO, sqrt_d, is_causal):
    B_k = 128
    N = K.shape[-2]

    # keep grads in the same dtype as inputs
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # ---- dtype fixes to keep Inductor happy ----
    inv_sqrt_d = Q.new_tensor(1.0 / sqrt_d)     # same dtype/device as Q
    L   = L.to(Q.dtype)
    O   = O.to(Q.dtype)
    dO  = dO.to(Q.dtype)
    # -------------------------------------------

    # D = sum_i dO_i * O_i (softmax backward identity)
    D = (O * dO).sum(dim=-1, keepdim=True)      # [..., N, 1]

    for j0 in range(0, N, B_k):
        j1 = min(j0 + B_k, N)
        K_j = K[..., j0:j1, :]
        V_j = V[..., j0:j1, :]

        # S: [..., N, Bk]
        S = einsum(Q, K_j, "... i d, ... j d -> ... i j") * inv_sqrt_d

        if is_causal:
            # mask strictly upper-triangular entries for this K-tile
            q_idx = torch.arange(S.shape[-2], device=S.device)
            k_idx = torch.arange(j0, j1, device=S.device)
            mask = (k_idx.unsqueeze(0) > q_idx.unsqueeze(1))  # [N, Bk]
            S = S.masked_fill(mask.unsqueeze(0), float("-inf"))

        # P using saved L for stability
        P = torch.exp(S - L.unsqueeze(-1))      # [..., N, Bk]

        # dV += P^T @ dO   (cast RHS to match destination dtype)
        dV[..., j0:j1, :] += einsum(P, dO, "... i j, ... i d -> ... j d").to(dV.dtype)

        # dS = P * (dP - D)  with dP = dO @ V_j^T
        dP = einsum(dO, V_j, "... i d, ... j d -> ... i j")
        dS = P * (dP - D)

        # dQ += dS @ K_j / sqrt_d
        dQ += (einsum(dS, K_j, "... a b, ... b c -> ... a c") * inv_sqrt_d).to(dQ.dtype)

        # dK[..., j0:j1, :] += dS^T @ Q / sqrt_d
        dK[..., j0:j1, :] += (einsum(dS, Q, "... a b, ... a d -> ... b d") * inv_sqrt_d).to(dK.dtype)

    return dQ, dK, dV, None

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q : torch.Tensor ,K : torch.Tensor,V : torch.Tensor, is_causal : tl.constexpr=False):
        B_q = B_k = 16
        # B_k = 1
        T_q = triton.cdiv(Q.shape[-2], B_q)
        d = Q.shape[-1]
        device = "cuda"
        # O_i = torch.empty(*Q.shape[:-2],B_q, d, device= device)
        O = torch.empty(Q.shape, device=Q.device)
        L = torch.empty(Q.shape[:-1], device=Q.device)
        batch_size = Q.shape[0]
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        scale = 1/math.sqrt(d)
        Q_TILE_SIZE = B_q
        K_TILE_SIZE = B_k
        D = d
        stride_qb = Q.stride(0)
        stride_qq = Q.stride(1)
        stride_qd = Q.stride(2)
        stride_kb = K.stride(0)
        stride_kk = K.stride(1)
        stride_kd = K.stride(2)
        stride_vb = V.stride(0)
        stride_vk = V.stride(1)
        stride_vd = V.stride(2)
        stride_ob = O.stride(0)
        stride_oq = O.stride(1)
        stride_od = O.stride(2)
        stride_lb = L.stride(0)
        stride_lq = L.stride(1)
        flash_fwd_kernel[(T_q, batch_size)](
            Q, K, V,
            O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
            )
        ctx.save_for_backward(Q,K,V,L,O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = math.sqrt(d)
        return O
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        return flash_attention_backward(Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal)



class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q : torch.Tensor ,K : torch.Tensor,V : torch.Tensor, is_causal=False):
        B_q = B_k = 16
        # B_k = 1
        T_q = math.ceil(Q.shape[-2] // B_q)
        T_k = math.ceil(K.shape[-2] // B_k)
        print(K.shape)
        d = Q.shape[-1]
        device = "cpu"
        O_i = torch.empty(*Q.shape[:-2],B_q, d, device= device)
        O = torch.empty(Q.shape, device=device)
        L = torch.empty(Q.shape[:-1], device=device)
        l_i = torch.empty(*Q.shape[:-2], B_q, device = device)
        m_i = torch.empty(*Q.shape[:-2], B_q, device = device)
        sqrt_d = math.sqrt(d)
        for i in range(T_q):
            offset_i = i*B_q
            Q_i = Q[:,offset_i:offset_i + B_q, :]
            O_i.zero_()
            l_i.zero_()
            m_i.fill_(float('-inf'))
            for j in range(T_k):
                offset_j = j*B_k
                K_j = K[:,offset_j:offset_j + B_k, :]
                V_j = V[:,offset_j:offset_j + B_k, :]
                S = einsum(Q_i, K_j, "... i d, ... j d -> ... i j")/sqrt_d
                m_i_new = torch.max(m_i, torch.max(S, dim = -1)[0])
                tildeP = torch.exp(S - m_i_new.unsqueeze(-1))
                l_i = torch.exp(m_i - m_i_new)*l_i + torch.sum(tildeP, dim=-1)
                O_i = einsum(torch.exp(m_i - m_i_new), O_i, "... a, ... a b-> ... a b") 
                O_i += einsum(tildeP, V_j, "... a b, ... b d -> ... a d")
                m_i = m_i_new
            O_i = einsum(l_i.reciprocal(), O_i, "... a, ... a b -> ... a b")
            O[:, offset_i:offset_i + B_q,:] = O_i
            L[:, offset_i:offset_i + B_q] = m_i + torch.log(l_i)
        ctx.save_for_backward(Q,K,V,L,O)
        ctx.is_causal = is_causal
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.sqrt_d = sqrt_d
        ctx.T_q = T_q
        ctx.T_k = T_k
        ctx.d = d
        return O
    
    def backward(ctx, dO):
        Q,K,V,L,O = ctx.saved_tensors
        D = (O * dO).sum(dim=-1, keepdim=True)
        S = einsum(Q, K, "... i d, ... j d -> ... i j")/ctx.sqrt_d
        P = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P, dO, "... i j, ... i d -> ... j d")
        dP = einsum(dO, V, "... i d, ... j d -> ... i j")
        dS = P * (dP - D)
        dQ = einsum(dS, K, "... a b, ... b c-> ... a c")/ctx.sqrt_d
        dK = einsum(dS, Q, "... a b, ... a d -> ... b d")/ctx.sqrt_d

        return dQ, dK, dV, None


