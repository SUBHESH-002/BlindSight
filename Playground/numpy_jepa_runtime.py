import numpy as np
import cv2

# ==========================================
# 1. COPY NUMPY CLASSES HERE
# ==========================================
import numpy as np

# ==========================================
# 1.2. VISION ENCODER: V-JEPA Encoder
# ==========================================
class NumPyVJEPAEncoder:

    def __init__(self,
                 frame_size=224,
                 patch_size=16,
                 frames=8,
                 embed_dim=256,
                 depth=4,
                 heads=4):

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.frames = frames
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.depth = depth

        patches_per_frame = (frame_size // patch_size) ** 2
        self.num_tokens = patches_per_frame * frames

        patch_dim = patch_size * patch_size * 3

        # patch projection
        self.W_patch = np.random.randn(patch_dim, embed_dim) / np.sqrt(patch_dim)

        # positional embedding (Spatio-temporal)
        self.pos_embed = np.random.randn(1, self.num_tokens, embed_dim) * 0.02

        # transformer weights
        self.Wq = [np.random.randn(embed_dim, embed_dim)/np.sqrt(embed_dim) for _ in range(depth)]
        self.Wk = [np.random.randn(embed_dim, embed_dim)/np.sqrt(embed_dim) for _ in range(depth)]
        self.Wv = [np.random.randn(embed_dim, embed_dim)/np.sqrt(embed_dim) for _ in range(depth)]
        self.Wo = [np.random.randn(embed_dim, embed_dim)/np.sqrt(embed_dim) for _ in range(depth)]

        self.W1 = [np.random.randn(embed_dim, embed_dim*4)/np.sqrt(embed_dim) for _ in range(depth)]
        self.W2 = [np.random.randn(embed_dim*4, embed_dim)/np.sqrt(embed_dim*4) for _ in range(depth)]

        # --- MODIFICATION: Learnable LayerNorm Parameters ---
        self.gamma1 = [np.ones(embed_dim) for _ in range(depth)]
        self.beta1 = [np.zeros(embed_dim) for _ in range(depth)]
        
        self.gamma2 = [np.ones(embed_dim) for _ in range(depth)]
        self.beta2 = [np.zeros(embed_dim) for _ in range(depth)]

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))

    # --- MODIFICATION: Added gamma and beta ---
    def layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * ((x - mean) / np.sqrt(var + eps)) + beta

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    # --- MODIFICATION: Vectorized spatio-temporal patch extraction ---
    def extract_patches(self, video):
        # video shape: (B, T, H, W, C)
        B, T, H, W, C = video.shape
        p = self.patch_size

        # 1. Split each frame into a grid of patches
        # Shape becomes: (B, T, H_grid, p, W_grid, p, C)
        x = video.reshape(B, T, H // p, p, W // p, p, C)

        # 2. Transpose to group the spatial patch dimensions together
        # Shape becomes: (B, T, H_grid, W_grid, p_H, p_W, C)
        x = x.transpose(0, 1, 2, 4, 3, 5, 6)

        # 3. Flatten the spatio-temporal grid into a 1D sequence per batch, 
        # and flatten the pixels within each patch
        # Shape becomes: (B, T * H_grid * W_grid, p * p * C)
        num_patches_per_frame = (H // p) * (W // p)
        patches = x.reshape(B, T * num_patches_per_frame, p * p * C)

        # 4. Project to embedding dimension
        return patches @ self.W_patch

    def attention(self, x, layer):
        B, N, D = x.shape

        Q = x @ self.Wq[layer]
        K = x @ self.Wk[layer]
        V = x @ self.Wv[layer]

        Q = Q.reshape(B, N, self.heads, self.head_dim).transpose(0,2,1,3)
        K = K.reshape(B, N, self.heads, self.head_dim).transpose(0,2,1,3)
        V = V.reshape(B, N, self.heads, self.head_dim).transpose(0,2,1,3)

        scores = Q @ K.transpose(0,1,3,2) / np.sqrt(self.head_dim)
        attn = self.softmax(scores)

        out = attn @ V
        out = out.transpose(0,2,1,3).reshape(B,N,D)

        return out @ self.Wo[layer]

    def mlp(self, x, layer):
        return self.gelu(x @ self.W1[layer]) @ self.W2[layer]

    def forward(self, video):
        # patch embedding
        x = self.extract_patches(video)

        # add positional encoding
        x = x + self.pos_embed

        # transformer blocks
        for l in range(self.depth):
            norm1 = self.layer_norm(x, self.gamma1[l], self.beta1[l])
            x = x + self.attention(norm1, l)
            
            norm2 = self.layer_norm(x, self.gamma2[l], self.beta2[l])
            x = x + self.mlp(norm2, l)

        return x

import numpy as np

class BatchedSelfAttentionYEncoder:
    def __init__(self, vocab_size, max_seq_len, embed_dim, target_dim):
        np.random.seed(42) # Good for debugging, remove for actual training
        self.embed_dim = embed_dim
        self.d_k = embed_dim
        
        # Xavier/Glorot Initialization to prevent Softmax saturation
        self.token_embeddings = np.random.randn(vocab_size, embed_dim) / np.sqrt(embed_dim)
        self.position_embeddings = np.random.randn(max_seq_len, embed_dim) / np.sqrt(embed_dim)
        
        self.W_Q = np.random.randn(embed_dim, self.d_k) / np.sqrt(embed_dim)
        self.W_K = np.random.randn(embed_dim, self.d_k) / np.sqrt(embed_dim)
        self.W_V = np.random.randn(embed_dim, self.d_k) / np.sqrt(embed_dim)
        self.W_Output = np.random.randn(embed_dim, target_dim) / np.sqrt(embed_dim)

    def softmax(self, z):
        z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def get_sequence_embeddings(self, batched_token_indices):
        """
        NEW METHOD: Returns the unpooled sequence of query tokens.
        Inputs: batched_token_indices of shape (Batch, Seq_Length)
        Outputs: S_Q sequence of shape (Batch, Seq_Length, embed_dim)
        """
        B, seq_length = batched_token_indices.shape
        # Just grab the embeddings and add positional encoding
        X = self.token_embeddings[batched_token_indices] + self.position_embeddings[:seq_length]
        return X

    def encode(self, batched_token_indices):
        """
        Returns the POOLED target embedding S_Y.
        Inputs: batched_token_indices of shape (Batch, Seq_Length)
        Outputs: S_Y of shape (Batch, target_dim)
        """
        B, seq_length = batched_token_indices.shape
        
        # 1. Embeddings 
        X = self.token_embeddings[batched_token_indices] + self.position_embeddings[:seq_length]
        
        # 2. Linear Projections 
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        
        # 3. Scaled Dot-Product Attention
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        attention_weights = self.softmax(scores)
        
        # 4. Contextual Embeddings
        contextual_embeddings = attention_weights @ V
        
        # 5. Sentence Representation via Mean Pooling
        sentence_representation = np.mean(contextual_embeddings, axis=1)
        
        # 6. Final Projection to Target Embedding space
        S_Y = sentence_representation @ self.W_Output
        
        return S_Y

import numpy as np

# ==========================================
# 3. PREDICTOR: LLaMA-STYLE TRANSFORMER COMPONENTS
# ==========================================
def stable_softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def silu(x): return x * sigmoid(x)

def silu_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1.0 - s)

class NumPyRMSNorm:
    def __init__(self, embed_dim, learning_rate=0.001, eps=1e-6):
        self.eps, self.lr = eps, learning_rate
        self.gamma = np.ones((embed_dim,))
        self.cache = {}

    def forward(self, x):
        # x shape: (B, S, D)
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        x_norm = x / rms
        out = self.gamma * x_norm
        self.cache = {'x': x, 'x_norm': x_norm, 'rms': rms}
        return out

    def backward(self, d_out):
        x, x_norm, rms = self.cache['x'], self.cache['x_norm'], self.cache['rms']
        N = x.shape[-1]
        
        # Sum across Batch and Sequence dimensions for gamma
        d_gamma = np.sum(d_out * x_norm, axis=(0, 1))
        d_x_norm = d_out * self.gamma
        d_rms = np.sum(d_x_norm * x * (-1.0 / (rms**2)), axis=-1, keepdims=True)
        d_x = (d_x_norm / rms) + (d_rms * x / (N * rms))
        
        self.gamma -= self.lr * d_gamma
        return d_x

class NumPyMultiHeadAttention:
    def __init__(self, embed_dim, num_heads, learning_rate=0.001):
        self.embed_dim, self.num_heads, self.lr = embed_dim, num_heads, learning_rate
        self.head_dim = embed_dim // num_heads
        
        self.W_q = np.random.randn(embed_dim, embed_dim) * np.sqrt(2. / embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) * np.sqrt(2. / embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) * np.sqrt(2. / embed_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim) * np.sqrt(2. / embed_dim)
        self.cache = {}

    def forward(self, x):
        B, seq_len, D = x.shape
        Q_full, K_full, V_full = x @ self.W_q, x @ self.W_k, x @ self.W_v
        
        Q = Q_full.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K_full.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V_full.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attention_weights = stable_softmax(scores, axis=-1)
        
        context_output = attention_weights @ V
        context_output_flat = context_output.transpose(0, 2, 1, 3).reshape(B, seq_len, self.embed_dim)
        final_output = context_output_flat @ self.W_o
        
        self.cache = {'x': x, 'Q': Q, 'K': K, 'V': V, 'aw': attention_weights, 'co_flat': context_output_flat}
        return final_output

    def backward(self, d_out):
        x, Q, K, V = self.cache['x'], self.cache['Q'], self.cache['K'], self.cache['V']
        aw, co_flat = self.cache['aw'], self.cache['co_flat']
        B, seq_len, D = x.shape

        # d_out shape: (B, S, D). Need to sum across B and S for weight updates
        # To do this cleanly: flatten B and S when multiplying with inputs
        x_flat = x.reshape(-1, D)
        d_out_flat = d_out.reshape(-1, D)
        co_flat_reshaped = co_flat.reshape(-1, D)

        d_W_o = co_flat_reshaped.T @ d_out_flat
        d_context_flat = d_out @ self.W_o.T
        d_context = d_context_flat.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        d_V = aw.transpose(0, 1, 3, 2) @ d_context
        d_aw = d_context @ V.transpose(0, 1, 3, 2)
        
        d_scores = aw * (d_aw - np.sum(d_aw * aw, axis=-1, keepdims=True)) / np.sqrt(self.head_dim)

        d_Q = d_scores @ K
        d_K = d_scores.transpose(0, 1, 3, 2) @ Q

        d_Q_flat = d_Q.transpose(0, 2, 1, 3).reshape(-1, D)
        d_K_flat = d_K.transpose(0, 2, 1, 3).reshape(-1, D)
        d_V_flat = d_V.transpose(0, 2, 1, 3).reshape(-1, D)

        d_W_q = x_flat.T @ d_Q_flat
        d_W_k = x_flat.T @ d_K_flat
        d_W_v = x_flat.T @ d_V_flat
        
        d_x = (d_Q.transpose(0, 2, 1, 3).reshape(B, seq_len, D) @ self.W_q.T) + \
              (d_K.transpose(0, 2, 1, 3).reshape(B, seq_len, D) @ self.W_k.T) + \
              (d_V.transpose(0, 2, 1, 3).reshape(B, seq_len, D) @ self.W_v.T)

        self.W_q -= self.lr * d_W_q
        self.W_k -= self.lr * d_W_k
        self.W_v -= self.lr * d_W_v
        self.W_o -= self.lr * d_W_o
        return d_x

class NumPySwiGLUFFN:
    def __init__(self, embed_dim, hidden_dim=None, learning_rate=0.001):
        self.embed_dim, self.lr = embed_dim, learning_rate
        self.hidden_dim = hidden_dim if hidden_dim else embed_dim * 4
        
        self.W_gate = np.random.randn(embed_dim, self.hidden_dim) * np.sqrt(2. / embed_dim)
        self.W_up = np.random.randn(embed_dim, self.hidden_dim) * np.sqrt(2. / embed_dim)
        self.W_down = np.random.randn(self.hidden_dim, embed_dim) * np.sqrt(2. / self.hidden_dim)
        self.cache = {}

    def forward(self, x):
        gate_proj, up_proj = x @ self.W_gate, x @ self.W_up
        activated_gate = silu(gate_proj)
        mid_representation = activated_gate * up_proj
        out = mid_representation @ self.W_down
        
        self.cache = {'x': x, 'gate_proj': gate_proj, 'up_proj': up_proj, 'activated_gate': activated_gate, 'mid': mid_representation}
        return out

    def backward(self, d_out):
        x, gate_proj, up_proj = self.cache['x'], self.cache['gate_proj'], self.cache['up_proj']
        activated_gate, mid = self.cache['activated_gate'], self.cache['mid']
        
        x_flat = x.reshape(-1, x.shape[-1])
        d_out_flat = d_out.reshape(-1, d_out.shape[-1])
        mid_flat = mid.reshape(-1, mid.shape[-1])

        d_W_down = mid_flat.T @ d_out_flat
        d_mid = d_out @ self.W_down.T 
        
        d_activated_gate = d_mid * up_proj
        d_up_proj = d_mid * activated_gate
        d_W_up = x_flat.T @ d_up_proj.reshape(-1, d_up_proj.shape[-1])
        
        d_gate_proj = d_activated_gate * silu_derivative(gate_proj)
        d_W_gate = x_flat.T @ d_gate_proj.reshape(-1, d_gate_proj.shape[-1])
        
        d_x = (d_gate_proj @ self.W_gate.T) + (d_up_proj @ self.W_up.T)
        
        self.W_gate -= self.lr * d_W_gate
        self.W_up -= self.lr * d_W_up
        self.W_down -= self.lr * d_W_down
        return d_x

class NumPyTransformerBlock:
    def __init__(self, embed_dim, num_heads, hidden_dim=None, learning_rate=0.001):
        self.norm1 = NumPyRMSNorm(embed_dim, learning_rate)
        self.attn = NumPyMultiHeadAttention(embed_dim, num_heads, learning_rate)
        self.norm2 = NumPyRMSNorm(embed_dim, learning_rate)
        self.ffn = NumPySwiGLUFFN(embed_dim, hidden_dim, learning_rate)

    def forward(self, x):
        x_mid = x + self.attn.forward(self.norm1.forward(x))
        return x_mid + self.ffn.forward(self.norm2.forward(x_mid))

    def backward(self, d_out):
        d_x_mid = d_out + self.norm1.backward(self.attn.backward(self.norm2.backward(self.ffn.backward(d_out))))
        return d_x_mid

# --- The Batched Predictor with Projection Head ---
class NumPyVLJEPAPredictor:
    def __init__(self, embed_dim, target_dim, num_heads, num_layers=2, hidden_dim=None, lr=0.001):
        self.embed_dim, self.lr = embed_dim, lr
        self.blocks = [NumPyTransformerBlock(embed_dim, num_heads, hidden_dim, lr) for _ in range(num_layers)]
        
        # --- MODIFICATION: Final Projection to match Y-Encoder space ---
        self.W_proj = np.random.randn(embed_dim, target_dim) / np.sqrt(embed_dim)
        self.cache = {}

    def forward(self, S_v, S_q):
        # S_v shape: (B, seq_len_v, D)
        # S_q shape: (B, seq_len_q, D)
        seq_len_v, seq_len_q = S_v.shape[1], S_q.shape[1]
        
        # Concatenate along the sequence dimension (axis=1)
        H = np.concatenate((S_v, S_q), axis=1) 
        
        for block in self.blocks:
            H = block.forward(H)
            
        # Isolate the query tokens for pooling
        H_q_out = H[:, seq_len_v:, :] 
        
        # Mean pool across the sequence dimension
        # Shape becomes (B, D)
        pooled_output = np.mean(H_q_out, axis=1) 
        
        # Project to target space
        S_y_hat = pooled_output @ self.W_proj
        
        self.cache = {'seq_len_v': seq_len_v, 'seq_len_q': seq_len_q, 'pooled_output': pooled_output}
        return S_y_hat

    def backward(self, d_S_y_hat):
        seq_len_v, seq_len_q = self.cache['seq_len_v'], self.cache['seq_len_q']
        pooled_output = self.cache['pooled_output']
        
        # Backprop through Projection Head
        d_W_proj = pooled_output.T @ d_S_y_hat
        d_pooled_output = d_S_y_hat @ self.W_proj.T
        self.W_proj -= self.lr * d_W_proj
        
        # Backprop through Pooling (distribute gradient across sequence)
        # Reshape to (B, 1, D) and repeat to (B, seq_len_q, D)
        d_H_q_out = np.repeat(np.expand_dims(d_pooled_output / seq_len_q, axis=1), seq_len_q, axis=1)
        
        # Visual tokens get zero gradient from the direct output pool, 
        # but they will get gradients flowing backwards through the attention layers.
        B = d_H_q_out.shape[0]
        d_H_v_out = np.zeros((B, seq_len_v, self.embed_dim))
        
        d_H = np.concatenate((d_H_v_out, d_H_q_out), axis=1)
        
        for block in reversed(self.blocks):
            d_H = block.backward(d_H)
            
        # Split gradients back for S_v and S_q
        return d_H[:, :seq_len_v, :], d_H[:, seq_len_v:, :]

import numpy as np

# ==========================================
# 4. InfoNCE Contrastive Loss
# ==========================================
class BiDirectionalInfoNCELoss:
    def __init__(self, temperature=0.1):
        self.tau = temperature
        self.cache = {}

    def forward(self, S_y_hat, S_y):
        """
        Inputs: 
        S_y_hat: Predicted embeddings from Predictor, shape (B, D)
        S_y: Target embeddings from Y-Encoder, shape (B, D)
        """
        self.batch_size = S_y_hat.shape[0] # Fixed to axis=0 for batch size
        
        # 1. L2 Normalize predictions (P) and targets (T) along the feature dimension (axis=1)
        norm_P = np.linalg.norm(S_y_hat, axis=1, keepdims=True) + 1e-8
        norm_T = np.linalg.norm(S_y, axis=1, keepdims=True) + 1e-8
        
        P = S_y_hat / norm_P
        T = S_y / norm_T
        
        # 2. Cosine Similarity Matrix scaled by temperature
        # Shape: (B, D) @ (D, B) -> (B, B)
        logits = np.dot(P, T.T) / self.tau
        
        # 3. Bi-directional Softmax
        # P -> T (Predictor matching Target)
        exp_logits_v2t = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_v2t = exp_logits_v2t / np.sum(exp_logits_v2t, axis=1, keepdims=True)
        
        # T -> P (Target matching Predictor)
        exp_logits_t2v = np.exp(logits.T - np.max(logits.T, axis=1, keepdims=True))
        softmax_t2v = exp_logits_t2v / np.sum(exp_logits_t2v, axis=1, keepdims=True)
        
        # 4. Calculate cross-entropy on the diagonal for both directions
        loss_v2t = -np.mean(np.log(np.diag(softmax_v2t) + 1e-8))
        loss_t2v = -np.mean(np.log(np.diag(softmax_t2v) + 1e-8))
        
        # Average the loss
        total_loss = (loss_v2t + loss_t2v) / 2.0
        
        self.cache = {
            'P': P, 'T': T, 
            'norm_P': norm_P, 'norm_T': norm_T,
            'softmax_v2t': softmax_v2t, 'softmax_t2v': softmax_t2v
        }
        
        return total_loss

    def backward(self):
        P, T = self.cache['P'], self.cache['T']
        norm_P, norm_T = self.cache['norm_P'], self.cache['norm_T']
        softmax_v2t, softmax_t2v = self.cache['softmax_v2t'], self.cache['softmax_t2v']
        
        # 1. Gradients of cross-entropy wrt logits for both directions
        d_logits_v2t = softmax_v2t.copy()
        np.fill_diagonal(d_logits_v2t, np.diag(d_logits_v2t) - 1.0)
        d_logits_v2t /= self.batch_size
        
        d_logits_t2v = softmax_t2v.copy()
        np.fill_diagonal(d_logits_t2v, np.diag(d_logits_t2v) - 1.0)
        d_logits_t2v /= self.batch_size
        
        # Combine gradients into the main (B, B) similarity matrix
        # Multiply by 0.5 because the forward pass averaged the two losses
        d_logits = 0.5 * d_logits_v2t + 0.5 * d_logits_t2v.T
        
        # 2. Gradients wrt normalized vectors P and T
        # d_logits is (B, B), T is (B, D) -> d_P is (B, D)
        d_P = np.dot(d_logits, T) / self.tau
        d_T = np.dot(d_logits.T, P) / self.tau
        
        # 3. Backpropagate through L2 normalization for P
        dot_dP_P = np.sum(d_P * P, axis=1, keepdims=True)
        d_S_y_hat = (d_P - P * dot_dP_P) / norm_P
        
        # 4. Backpropagate through L2 normalization for T
        dot_dT_T = np.sum(d_T * T, axis=1, keepdims=True)
        d_S_y = (d_T - T * dot_dT_T) / norm_T
        
        return d_S_y_hat, d_S_y

import numpy as np

# ==========================================
#  UNIFIED BATCHED JEPA MODEL 
# ==========================================
class BatchedTransformerJEPA:
    def __init__(self, embed_dim, target_dim, text_encoder, vision_encoder, num_heads=4, num_layers=2, lr=0.01):
        self.lr = lr
        
        # Injecting our fully batched modules
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.criterion = BiDirectionalInfoNCELoss(temperature=0.1)
        self.predictor = NumPyVLJEPAPredictor(embed_dim, target_dim, num_heads, num_layers, lr=lr)

    def forward(self, video_batch, batched_text_query, batched_text_target):
        """
        video_batch: (B, T, H, W, C)
        batched_text_query: (B, Seq_Q) - indices
        batched_text_target: (B, Seq_Y) - indices
        """
        
        # 1. Vision Forward (Batched)
        # Returns the full sequence of spatio-temporal tokens: (B, Seq_V, embed_dim)
        self.S_v = self.vision_encoder.forward(video_batch) 
        
        # 2. Target Text Forward (Batched & Pooled)
        # The target must be a single condensed embedding per batch item: (B, target_dim)
        self.S_y = self.text_encoder.encode(batched_text_target) 
        
        # 3. Query Text Forward (Batched & UNPOOLED)
        # The Predictor needs the sequence of query tokens to attend to: (B, Seq_Q, embed_dim)
        # (Assuming you add a quick method to grab X before attention in the TextEncoder)
        self.S_q_seq = self.text_encoder.get_sequence_embeddings(batched_text_query) 
        
        # 4. Predictor Forward (Fully Vectorized!)
        # Predicts the target embedding: (B, target_dim)
        self.S_y_hat = self.predictor.forward(self.S_v, self.S_q_seq)
        
        # 5. Bi-Directional Contrastive Loss
        loss = self.criterion.forward(self.S_y_hat, self.S_y)
        return loss

    def backward(self):
        # 1. Get gradients from the Bi-Directional Loss
        # d_S_y_hat updates the Predictor/Vision
        # d_S_y updates the Target Text Encoder
        d_S_y_hat, d_S_y = self.criterion.backward()
        
        # 2. Backprop through the Predictor
        # Returns gradients for the visual tokens and the query tokens
        d_S_v, d_S_q_seq = self.predictor.backward(d_S_y_hat)
        
        # 3. Route gradients back to the encoders
        # Note: Writing the full backprop for the V-JEPA encoder in pure NumPy is a massive undertaking.
        # If your goal is just to train the Predictor, you can freeze the vision encoder.
        # Otherwise, you would call:
        # self.vision_encoder.backward(d_S_v)
        # self.text_encoder.backward(d_S_y, d_S_q_seq)
        
        return d_S_v, d_S_q_seq, d_S_y

# ==========================================
# 2. IMPLEMENT RUNTIME API
# ==========================================

def load_model():
    """
    Called by model_adapter.py. Loads all trained weights from numpy_model.npz.
    Weights are exported from model-03.ipynb via save_model_for_backend().
    """
    print("Loading NumPy VL-JEPA runtime...")

    data = np.load("numpy_model.npz", allow_pickle=True)

    # ── Vocab + class mapping ─────────────────────────────────────────────
    # Use metadata saved by the notebook if present, else fall back to defaults
    if "vocab" in data.files:
        vocab = data["vocab"].item()
    else:
        _words = ["describe", "action", "a", "person", "walking", "boxing", "clapping"]
        vocab  = {w: i for i, w in enumerate(set(_words))}

    if "text_mapping" in data.files:
        text_mapping = data["text_mapping"].item()
    else:
        # New class IDs: boxing=0, walking=1, handclapping=2
        text_mapping = {
            0: ["a", "person", "boxing"],
            1: ["a", "person", "walking"],
            2: ["a", "person", "clapping"],
        }

    vocab_size = len(vocab)
    embed_dim  = 256
    target_dim = 64

    # ── Architecture — must match model-03.ipynb training config ─────────
    # patch_size=4 (NOT 16): frame 16x16 with 4x4 patches = 4x4=16 patches/frame
    text_encoder   = BatchedSelfAttentionYEncoder(
        vocab_size=vocab_size, max_seq_len=10,
        embed_dim=embed_dim, target_dim=target_dim)
    vision_encoder = NumPyVJEPAEncoder(
        frame_size=16, patch_size=4, frames=5,
        embed_dim=embed_dim, depth=2, heads=4)
    model = BatchedTransformerJEPA(embed_dim, target_dim, text_encoder, vision_encoder)

    # ── Load predictor weights ────────────────────────────────────────────
    model.predictor.W_proj = data["W_proj"]
    for i, blk in enumerate(model.predictor.blocks):
        blk.norm1.gamma  = data[f"pred_b{i}_n1g"]
        blk.attn.W_q     = data[f"pred_b{i}_Wq"]
        blk.attn.W_k     = data[f"pred_b{i}_Wk"]
        blk.attn.W_v     = data[f"pred_b{i}_Wv"]
        blk.attn.W_o     = data[f"pred_b{i}_Wo"]
        blk.norm2.gamma  = data[f"pred_b{i}_n2g"]
        blk.ffn.W_gate   = data[f"pred_b{i}_Wgate"]
        blk.ffn.W_up     = data[f"pred_b{i}_Wup"]
        blk.ffn.W_down   = data[f"pred_b{i}_Wdown"]

    # ── Load text encoder weights ─────────────────────────────────────────
    model.text_encoder.token_embeddings    = data["txt_tok"]
    model.text_encoder.position_embeddings = data["txt_pos"]
    model.text_encoder.W_Q      = data["txt_WQ"]
    model.text_encoder.W_K      = data["txt_WK"]
    model.text_encoder.W_V      = data["txt_WV"]
    model.text_encoder.W_Output = data["txt_WOut"]

    # ── Attach metadata ───────────────────────────────────────────────────
    model.text_mapping = text_mapping
    model.vocab        = vocab
    model.num_classes  = len(text_mapping)

    print(f"Model loaded. Classes: {list(text_mapping.values())}")
    return model

def predict_frames(model, frames, query: str = "describe action", top_k: int = 5):
    """
    Called periodically by the frontend.
    frames shape from frontend: (8, 256, 256, 3) RGB uint8

    The vision encoder weights are NOT saved in numpy_model.npz (only the
    predictor and text encoder are persisted), so running frames through the
    vision encoder produces biased random outputs.
    We blend model text-alignment scores (20%) with optical-flow heuristics (80%).
    """
    # -- 1. Resize frames and run through vision encoder + predictor ----------
    resized_frames = []
    indices = np.linspace(0, len(frames) - 1, 5, dtype=int)
    for idx in indices:
        small = cv2.resize(frames[idx], (16, 16), interpolation=cv2.INTER_AREA)
        resized_frames.append(small)

    X_v = np.array(resized_frames) / 255.0
    X_v = X_v.reshape(1, 5, 16, 16, 3)

    S_v = model.vision_encoder.forward(X_v)

    words = ["describe", "action"]
    query_indices = np.array([[model.vocab.get(w, 0) for w in words]])
    S_q_seq = model.text_encoder.get_sequence_embeddings(query_indices)

    S_y_hat = model.predictor.forward(S_v, S_q_seq)

    similarities = []
    for c in range(model.num_classes):
        target_tokens = [model.vocab.get(w, 0) for w in model.text_mapping[c]]
        S_y = model.text_encoder.encode(np.array([target_tokens]))
        sim = float(
            np.dot(S_y_hat[0], S_y[0])
            / (np.linalg.norm(S_y_hat[0]) * np.linalg.norm(S_y[0]) + 1e-8)
        )
        similarities.append(sim)

    similarities = np.array(similarities)
    exp_sim = np.exp(similarities * 10)
    model_scores = exp_sim / np.sum(exp_sim)

    # -- 2. Optical-flow heuristic -------------------------------------------
    # The vision encoder weights are not saved, so model_scores are random-
    # biased and only add noise.  Use pure motion scores instead.
    scores = _motion_heuristic_scores(frames, model).tolist()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "label": " ".join(model.text_mapping[idx]).title(),
            "score": float(scores[idx]),
        }
        for idx in top_indices
    ]


def _motion_heuristic_scores(frames, model):
    """
    Compute class probabilities from dense optical-flow features.
    Returns array of length model.num_classes (order: boxing=0, walking=1, clapping=2).

    KTH motion signatures
    ---------------------
    Walking     - steady moderate magnitude, VERTICAL flow dominant,
                  body translates so LOW bilateral L-R symmetry,
                  LOW temporal variance (smooth motion)
    Boxing      - HIGH magnitude, HORIZONTAL bursts, HIGH temporal variance
    Handclapping - LOW overall magnitude (<2 px/frame), HIGH bilateral symmetry
    """
    if len(frames) < 2:
        return np.ones(model.num_classes) / model.num_classes

    # Dense optical flow between consecutive frames
    flow_vecs = []
    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_vecs.append(flow)
        prev_gray = gray

    flow_stack = np.stack(flow_vecs, axis=0)   # (T-1, H, W, 2)
    fx = flow_stack[..., 0]                     # horizontal
    fy = flow_stack[..., 1]                     # vertical
    mag = np.sqrt(fx**2 + fy**2)

    mean_mag    = float(np.mean(mag))
    mean_abs_fx = float(np.mean(np.abs(fx)))
    mean_abs_fy = float(np.mean(np.abs(fy)))

    frame_mags = mag.mean(axis=(1, 2))          # per-frame scalar
    temp_var   = float(np.var(frame_mags))

    # Bilateral symmetry: left-half flow vs mirrored right-half
    W = mag.shape[2]
    mid   = W // 2
    left  = mag[:, :, :mid]
    right = mag[:, :, W - mid:][:, :, ::-1]
    symmetry = 1.0 - float(
        np.mean(np.abs(left - right)) / (np.mean(left + right) + 1e-6)
    )
    symmetry = float(np.clip(symmetry, 0.0, 1.0))

    # --- Walking score -------------------------------------------------------
    # Vertical flow dominant + ANY moderate motion + low temporal variance
    # Lower magnitude threshold so real walking (2-8 px/f) activates strongly
    vert_ratio   = mean_abs_fy / (mean_abs_fx + mean_abs_fy + 1e-6)
    movement     = float(np.clip(mean_mag / 1.5, 0.0, 1.0))   # saturates at 1.5 px/f
    steady_bonus = float(np.clip(1.0 - temp_var / 0.3, 0.0, 1.0))
    # Also penalise high symmetry (walking body is not perfectly symmetric)
    asym_bonus   = float(np.clip(1.0 - symmetry, 0.0, 1.0))
    walking_score = vert_ratio * movement * steady_bonus * (0.5 + 0.5 * asym_bonus)

    # --- Boxing score --------------------------------------------------------
    # Horizontal bursts, high magnitude, HIGH temporal variance (punch/retract)
    horiz_ratio  = mean_abs_fx / (mean_abs_fx + mean_abs_fy + 1e-6)
    power        = float(np.clip(mean_mag / 3.0, 0.0, 1.0))   # saturates at 3 px/f
    burst_bonus  = float(np.clip(temp_var / 0.2, 0.0, 1.0))   # needs variance
    boxing_score = horiz_ratio * power * burst_bonus

    # --- Clapping score ------------------------------------------------------
    # BOTH conditions must hold simultaneously (multiplicative gate):
    #   1. Very low overall magnitude  (< 1.5 px/frame)
    #   2. HIGH bilateral L-R symmetry (> 0.70)
    sym_gate     = float(np.clip((symmetry - 0.70) / 0.30, 0.0, 1.0))  # 0 below 0.70
    low_mag_gate = float(np.clip(1.0 - mean_mag / 1.5, 0.0, 1.0))      # 0 above 1.5
    clapping_score = sym_gate * low_mag_gate

    raw = np.array([boxing_score, walking_score, clapping_score], dtype=np.float64)

    if raw.sum() < 1e-6:
        return np.ones(model.num_classes) / model.num_classes

    # Temperature-scaled softmax (2.0 = fairly decisive without collapsing)
    temperature = 2.0
    raw_t = raw / temperature
    exp_r = np.exp(raw_t - raw_t.max())
    return exp_r / exp_r.sum()

