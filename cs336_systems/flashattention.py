import torch
import math
from einops import rearrange, einsum


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
        return O
    
    def backward():
        raise NotImplementedError
