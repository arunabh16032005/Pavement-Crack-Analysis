"""
Script to generate publication-quality architecture diagram for PROJ1.
Outputs:
1. architecture_diagram.svg (Vector format for papers/presentations)
2. architecture_diagram.png (High-resolution rasterized image, 2420x1040)
"""

import os
import base64
import subprocess
from pathlib import Path

def get_base64_img(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            ext = os.path.splitext(img_path)[1].replace(".", "")
            if ext == "jpg": ext = "jpeg"
            return f"data:image/{ext};base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

def generate_svg():
    assets_dir = Path(r"c:\Users\Arunabh\Desktop\PROJ1\diagram_assets")
    
    raw_crack_b64 = get_base64_img(assets_dir / "raw_crack.png")
    prep_token_b64 = get_base64_img(assets_dir / "prep_token.png")
    frangi_b64 = get_base64_img(assets_dir / "frangi.png")
    wavelet_b64 = get_base64_img(assets_dir / "wavelet.png")
    mamba_b64 = get_base64_img(assets_dir / "mamba.png")
    cross_b64 = get_base64_img(assets_dir / "cross_attn.png")
    vqc_b64 = get_base64_img(assets_dir / "vqc.png")
    structural_b64 = get_base64_img(assets_dir / "structural.png")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2420 1040" width="2420" height="1040" style="background:#ffffff; font-family:'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Filters & Shadows -->
    <filter id="card-shadow" x="-4%" y="-3%" width="108%" height="108%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#000000" flood-opacity="0.06"/>
    </filter>
    <filter id="block-shadow" x="-6%" y="-6%" width="112%" height="112%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#000000" flood-opacity="0.09"/>
    </filter>
    
    <!-- Gradients -->
    <linearGradient id="grad-input" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#f1f5f9"/>
    </linearGradient>
    <linearGradient id="grad-frozen" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#eff6ff"/>
      <stop offset="100%" stop-color="#dbeafe"/>
    </linearGradient>
    <linearGradient id="grad-quantum" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f0fdf4"/>
      <stop offset="100%" stop-color="#dcfce7"/>
    </linearGradient>
    <linearGradient id="grad-backbone" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#faf5ff"/>
      <stop offset="100%" stop-color="#f3e8ff"/>
    </linearGradient>
    <linearGradient id="grad-refine" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fff1f2"/>
      <stop offset="100%" stop-color="#ffe4e6"/>
    </linearGradient>
    <linearGradient id="grad-struct" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fffbeb"/>
      <stop offset="100%" stop-color="#fef3c7"/>
    </linearGradient>
    <linearGradient id="grad-severity" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fff7ed"/>
      <stop offset="100%" stop-color="#ffedd5"/>
    </linearGradient>
    <linearGradient id="grad-outputs" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#e2e8f0"/>
    </linearGradient>
    <linearGradient id="grad-legend" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f8fafc"/>
    </linearGradient>
    <linearGradient id="grad-optim" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f0f9ff"/>
      <stop offset="100%" stop-color="#e0f2fe"/>
    </linearGradient>

    <!-- Markers -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1e293b"/>
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb"/>
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#7c3aed"/>
    </marker>
    <marker id="arrow-dashed" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0284c7"/>
    </marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#ea580c"/>
    </marker>
    <marker id="arrow-rose" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#e11d48"/>
    </marker>
  </defs>

  <!-- ==================== MAIN HEADER ==================== -->
  <g id="header">
    <text x="1210" y="38" font-size="26" font-weight="800" text-anchor="middle" fill="#0f172a" letter-spacing="0.5">QUANTUM-ENHANCED PAVEMENT CONDITION MONITORING SYSTEM</text>
    <text x="1210" y="60" font-size="14" font-weight="500" text-anchor="middle" fill="#475569">End-to-End Hybrid Quantum-Classical Framework for Automated Crack Segmentation, Severity Prediction, and Pavement Assessment</text>
  </g>

  <!-- ==================== 1. INPUT & PREPROCESSING ==================== -->
  <g id="panel-input" transform="translate(25, 80)">
    <rect width="215" height="740" rx="12" fill="url(#grad-input)" stroke="#cbd5e1" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="215" height="40" rx="12" fill="#e2e8f0"/>
    <rect x="0" y="20" width="215" height="20" fill="#e2e8f0"/>
    <text x="107" y="26" font-size="13" font-weight="800" text-anchor="middle" fill="#0f172a">INPUT &amp; PREPROCESSING</text>

    <!-- Raw Image Thumbnail -->
    <g transform="translate(37, 52)">
      <rect width="140" height="110" rx="8" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5"/>
      <image href="{raw_crack_b64}" x="10" y="8" width="120" height="74" preserveAspectRatio="xMidYMid slice"/>
      <text x="70" y="96" font-size="10.5" font-weight="700" text-anchor="middle" fill="#334155">Raw Pavement Image</text>
      <text x="70" y="107" font-size="9" font-weight="600" text-anchor="middle" fill="#64748b">I_raw ∈ ℝ^(H×W×3)</text>
    </g>

    <!-- Down Arrow -->
    <path d="M 107 175 L 107 205" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>
    <text x="107" y="195" font-size="9" font-weight="700" text-anchor="middle" fill="#475569" dx="42">Raw Stream</text>

    <!-- Preprocessing Pipeline Box -->
    <g transform="translate(18, 215)">
      <rect width="178" height="495" rx="10" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5"/>
      <text x="89" y="24" font-size="12" font-weight="800" text-anchor="middle" fill="#1e293b">Preprocessing Pipeline</text>

      <!-- Step 1: Standardization -->
      <g transform="translate(14, 36)">
        <rect width="150" height="76" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
        <text x="75" y="18" font-size="11" font-weight="700" text-anchor="middle" fill="#0f172a">1. Standardization</text>
        <text x="75" y="35" font-size="9.5" text-anchor="middle" fill="#475569">Resize to 512×512</text>
        <text x="75" y="50" font-size="9.5" text-anchor="middle" fill="#475569">Normalize [0, 1] float32</text>
        <text x="75" y="64" font-size="8.5" text-anchor="middle" fill="#64748b">ImageNet μ, σ stats</text>
      </g>

      <!-- Step 2: Noise Suppression -->
      <g transform="translate(14, 122)">
        <rect width="150" height="76" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
        <text x="75" y="18" font-size="11" font-weight="700" text-anchor="middle" fill="#0f172a">2. Noise Suppression</text>
        <text x="75" y="35" font-size="9.5" text-anchor="middle" fill="#475569">Non-Local Means (h=10)</text>
        <text x="75" y="50" font-size="9.5" text-anchor="middle" fill="#475569">Bilateral Filter (d=9)</text>
        <text x="75" y="64" font-size="8.5" text-anchor="middle" fill="#64748b">Edge-preserving smoothing</text>
      </g>

      <!-- Step 3: CLAHE -->
      <g transform="translate(14, 208)">
        <rect width="150" height="76" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
        <text x="75" y="18" font-size="11" font-weight="700" text-anchor="middle" fill="#0f172a">3. CLAHE Enhancement</text>
        <text x="75" y="35" font-size="9.5" text-anchor="middle" fill="#475569">L-Channel in LAB Space</text>
        <text x="75" y="50" font-size="9.5" text-anchor="middle" fill="#475569">Clip Limit=2.0, Tile 8×8</text>
        <text x="75" y="64" font-size="8.5" text-anchor="middle" fill="#64748b">Illumination balancing</text>
      </g>

      <!-- Step 4: Augmentation -->
      <g transform="translate(14, 294)">
        <rect width="150" height="76" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
        <text x="75" y="18" font-size="11" font-weight="700" text-anchor="middle" fill="#0f172a">4. Data Augmentation</text>
        <text x="75" y="35" font-size="9.5" text-anchor="middle" fill="#475569">Elastic Transform (α=120)</text>
        <text x="75" y="50" font-size="9.5" text-anchor="middle" fill="#475569">Rot ±15°, Flips, Jitter</text>
        <text x="75" y="64" font-size="8.5" text-anchor="middle" fill="#64748b">Joint Mask Transformation</text>
      </g>

      <!-- Step 5: ROI Localization -->
      <g transform="translate(14, 380)">
        <rect width="150" height="95" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
        <text x="75" y="18" font-size="11" font-weight="700" text-anchor="middle" fill="#0f172a">5. ROI Localization</text>
        <text x="75" y="35" font-size="9.5" text-anchor="middle" fill="#475569">GrabCut + Otsu Pavement</text>
        <text x="75" y="50" font-size="9.5" text-anchor="middle" fill="#475569">Non-pavement masking</text>
        <text x="75" y="68" font-size="9.5" font-weight="700" text-anchor="middle" fill="#2563eb">Output: I_pre (512×512)</text>
        <text x="75" y="82" font-size="8.5" text-anchor="middle" fill="#64748b">Standardized Token</text>
      </g>
    </g>
  </g>

  <!-- Transition Arrow: Input to Encoders -->
  <g transform="translate(240, 500)">
    <path d="M 0 0 L 55 0" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)"/>
    <g transform="translate(12, -45)">
      <rect width="30" height="30" rx="4" fill="#ffffff" stroke="#64748b" stroke-width="1.2"/>
      <image href="{prep_token_b64}" x="3" y="3" width="24" height="24" preserveAspectRatio="xMidYMid slice"/>
    </g>
    <text x="27" y="18" font-size="9.5" font-weight="700" text-anchor="middle" fill="#1e293b">Preprocessed Token</text>
    <text x="27" y="32" font-size="9" font-weight="600" text-anchor="middle" fill="#64748b">I_pre (512×512×3)</text>
  </g>

  <!-- ==================== 2. FROZEN PRE-TRAINED ENCODERS ==================== -->
  <g id="panel-encoders" transform="translate(305, 80)">
    <rect width="200" height="740" rx="12" fill="url(#grad-frozen)" stroke="#93c5fd" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="200" height="40" rx="12" fill="#bfdbfe"/>
    <rect x="0" y="20" width="200" height="20" fill="#bfdbfe"/>
    <!-- NO EMOJI! Clean typography -->
    <text x="100" y="18" font-size="13" font-weight="800" text-anchor="middle" fill="#1e3a8a">FROZEN PRE-TRAINED</text>
    <text x="100" y="33" font-size="12.5" font-weight="800" text-anchor="middle" fill="#1e3a8a">ENCODERS</text>

    <!-- Branch 1: Frangi / Hessian -->
    <g transform="translate(15, 50)">
      <rect width="170" height="135" rx="8" fill="#ffffff" stroke="#60a5fa" stroke-width="1.5"/>
      <image href="{frangi_b64}" x="12" y="10" width="55" height="55" preserveAspectRatio="xMidYMid slice"/>
      <text x="75" y="26" font-size="11" font-weight="700" fill="#1e3a8a">Frangi / Hessian</text>
      <text x="75" y="42" font-size="9.5" fill="#2563eb">Multi-Scale σ={1,2,4,8}</text>
      <text x="75" y="56" font-size="9" fill="#64748b">Eigenvalue Ratio λ₁/λ₂</text>
      <line x1="12" y1="74" x2="158" y2="74" stroke="#e2e8f0" stroke-width="1"/>
      <text x="85" y="92" font-size="10" font-weight="700" text-anchor="middle" fill="#0369a1">Curvilinear Ridge Map</text>
      <text x="85" y="110" font-size="10" text-anchor="middle" fill="#0284c7" font-weight="800">F_curv ∈ ℝ^(H×W×C₁)</text>
      <text x="85" y="124" font-size="8.5" text-anchor="middle" fill="#64748b">Tube-like Crack Geometry</text>
    </g>

    <!-- Branch 2: Wavelet / Gabor -->
    <g transform="translate(15, 200)">
      <rect width="170" height="135" rx="8" fill="#ffffff" stroke="#60a5fa" stroke-width="1.5"/>
      <image href="{wavelet_b64}" x="12" y="10" width="55" height="55" preserveAspectRatio="xMidYMid slice"/>
      <text x="75" y="26" font-size="11" font-weight="700" fill="#1e3a8a">Wavelet / Gabor</text>
      <text x="75" y="42" font-size="9.5" fill="#2563eb">Morlet J=3, L=8</text>
      <text x="75" y="56" font-size="9" fill="#64748b">8 Orients × 5 Freqs</text>
      <line x1="12" y1="74" x2="158" y2="74" stroke="#e2e8f0" stroke-width="1"/>
      <text x="85" y="92" font-size="10" font-weight="700" text-anchor="middle" fill="#0369a1">Spectral-Texture Bank</text>
      <text x="85" y="110" font-size="10" text-anchor="middle" fill="#0284c7" font-weight="800">F_tex ∈ ℝ^(H×W×C₂)</text>
      <text x="85" y="124" font-size="8.5" text-anchor="middle" fill="#64748b">Persistent Homology (H₀, H₁)</text>
    </g>

    <!-- Branch 3: Vision Mamba (SSM) -->
    <g transform="translate(15, 350)">
      <rect width="170" height="142" rx="8" fill="#ffffff" stroke="#60a5fa" stroke-width="1.5"/>
      <image href="{mamba_b64}" x="12" y="10" width="55" height="55" preserveAspectRatio="xMidYMid slice"/>
      <text x="75" y="26" font-size="11" font-weight="700" fill="#1e3a8a">Vision Mamba</text>
      <text x="75" y="42" font-size="9.5" fill="#2563eb">SS2D Bi-directional</text>
      <text x="75" y="56" font-size="9" fill="#64748b">Selective S6 Model</text>
      <line x1="12" y1="74" x2="158" y2="74" stroke="#e2e8f0" stroke-width="1"/>
      <text x="85" y="92" font-size="10" font-weight="700" text-anchor="middle" fill="#0369a1">Semantic State Space</text>
      <text x="85" y="110" font-size="10" text-anchor="middle" fill="#0284c7" font-weight="800">F_sem ∈ ℝ^(H×W×C₃)</text>
      <text x="85" y="124" font-size="8.5" text-anchor="middle" fill="#64748b">O(N) Linear Global Context</text>
      <text x="85" y="136" font-size="8" text-anchor="middle" fill="#64748b">Pre-trained ImageNet-1K</text>
    </g>

    <!-- Fusion: Cross-Attention Module -->
    <g transform="translate(15, 508)">
      <rect width="170" height="215" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2"/>
      <image href="{cross_b64}" x="58" y="8" width="55" height="50" preserveAspectRatio="xMidYMid slice"/>
      <text x="85" y="74" font-size="11.5" font-weight="800" text-anchor="middle" fill="#1e3a8a">Cross-Attention Fusion</text>
      <text x="85" y="90" font-size="9.5" text-anchor="middle" fill="#2563eb">Query-Key Interaction</text>
      <text x="85" y="104" font-size="9" text-anchor="middle" fill="#64748b">Softmax(QK^T / √d_k) · V</text>
      <rect x="12" y="114" width="146" height="42" rx="4" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1"/>
      <text x="85" y="130" font-size="9.5" font-weight="700" text-anchor="middle" fill="#0369a1">SE-Net Channel Gate</text>
      <text x="85" y="144" font-size="8.5" text-anchor="middle" fill="#0284c7">Adaptive Weighting</text>
      <text x="85" y="174" font-size="11" font-weight="800" text-anchor="middle" fill="#1e293b">Multi-Domain Token</text>
      <text x="85" y="192" font-size="10.5" font-weight="800" text-anchor="middle" fill="#2563eb">F_c ∈ ℝ^(H×W×C_f)</text>
    </g>

    <!-- Internal Vertical Connectors -->
    <path d="M 100 185 L 100 200" stroke="#3b82f6" stroke-width="1.8" marker-end="url(#arrow-blue)"/>
    <path d="M 100 335 L 100 350" stroke="#3b82f6" stroke-width="1.8" marker-end="url(#arrow-blue)"/>
    <path d="M 100 492 L 100 508" stroke="#3b82f6" stroke-width="1.8" marker-end="url(#arrow-blue)"/>
  </g>

  <!-- Transition Arrow: Encoders to Quantum Module -->
  <g transform="translate(505, 620)">
    <path d="M 0 0 L 55 0" stroke="#1e293b" stroke-width="2.5" marker-end="url(#arrow)"/>
    <text x="27" y="-12" font-size="9.5" font-weight="800" text-anchor="middle" fill="#0f172a">F_c Token</text>
    <text x="27" y="16" font-size="8.5" font-weight="600" text-anchor="middle" fill="#64748b">C_f Features</text>
  </g>

  <!-- ==================== 3. QUANTUM FEATURE MODULE ==================== -->
  <g id="panel-quantum" transform="translate(565, 80)">
    <rect width="195" height="740" rx="12" fill="url(#grad-quantum)" stroke="#86efac" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="195" height="40" rx="12" fill="#bbf7d0"/>
    <rect x="0" y="20" width="195" height="20" fill="#bbf7d0"/>
    <!-- NO EMOJI! Clean typography -->
    <text x="97" y="18" font-size="13" font-weight="800" text-anchor="middle" fill="#14532d">QUANTUM FEATURE</text>
    <text x="97" y="33" font-size="12.5" font-weight="800" text-anchor="middle" fill="#14532d">MODULE</text>

    <!-- PennyLane Badge -->
    <g transform="translate(32, 50)">
      <rect width="130" height="32" rx="16" fill="#ffffff" stroke="#16a34a" stroke-width="1.5"/>
      <circle cx="20" cy="16" r="10" fill="#dcfce7"/>
      <text x="20" y="20" font-size="11" font-weight="800" text-anchor="middle" fill="#15803d">PL</text>
      <text x="75" y="21" font-size="12" font-weight="800" fill="#166534">PennyLane</text>
    </g>

    <!-- Linear Projection Sub-block -->
    <g transform="translate(15, 94)">
      <rect width="165" height="66" rx="6" fill="#ffffff" stroke="#86efac" stroke-width="1.2"/>
      <text x="82" y="19" font-size="11" font-weight="700" text-anchor="middle" fill="#166534">Linear Projection</text>
      <text x="82" y="36" font-size="9.5" text-anchor="middle" fill="#475569">Dimension Compression</text>
      <text x="82" y="52" font-size="10.5" font-weight="700" text-anchor="middle" fill="#15803d">C_f → n_qubits (8)</text>
    </g>

    <!-- Parameterized Angle Encoding -->
    <g transform="translate(15, 172)">
      <rect width="165" height="88" rx="6" fill="#ffffff" stroke="#86efac" stroke-width="1.2"/>
      <text x="82" y="19" font-size="11" font-weight="700" text-anchor="middle" fill="#166534">Angle Encoding</text>
      <text x="82" y="38" font-size="10" font-weight="700" text-anchor="middle" fill="#15803d">|x_i⟩ = RY(θ_i·x_i)|0⟩</text>
      <text x="82" y="54" font-size="9" text-anchor="middle" fill="#64748b">Feature-to-State Mapping</text>
      <rect x="18" y="62" width="130" height="18" rx="4" fill="#f0fdf4"/>
      <text x="82" y="75" font-size="8.5" font-weight="600" text-anchor="middle" fill="#166534">Trainable Scaling θ_i</text>
    </g>

    <!-- VQC Circuit Diagram -->
    <g transform="translate(15, 272)">
      <rect width="165" height="235" rx="8" fill="#ffffff" stroke="#22c55e" stroke-width="1.5"/>
      <text x="82" y="20" font-size="11.5" font-weight="800" text-anchor="middle" fill="#15803d">Variational Circuit (VQC)</text>
      <image href="{vqc_b64}" x="15" y="28" width="135" height="82" preserveAspectRatio="xMidYMid contain"/>
      <line x1="12" y1="118" x2="153" y2="118" stroke="#e2e8f0" stroke-width="1"/>
      <text x="82" y="136" font-size="10.5" font-weight="700" text-anchor="middle" fill="#0f172a">Circuit Architecture</text>
      <text x="82" y="152" font-size="9.5" text-anchor="middle" fill="#475569">• Rotations: RY(θ) ⊗ RZ(φ)</text>
      <text x="82" y="168" font-size="9.5" text-anchor="middle" fill="#475569">• CNOT Entanglement Chain</text>
      <text x="82" y="184" font-size="9.5" text-anchor="middle" fill="#475569">• Circuit Depth: D = 4–8</text>
      <text x="82" y="200" font-size="9.5" text-anchor="middle" fill="#475569">• Pauli-Z Measurement ⟨σ_z⟩</text>
      <text x="82" y="220" font-size="9.5" font-weight="800" text-anchor="middle" fill="#15803d">default.qubit Simulator</text>
    </g>

    <!-- Output Token Card -->
    <g transform="translate(15, 520)">
      <rect width="165" height="205" rx="8" fill="#ffffff" stroke="#16a34a" stroke-width="2"/>
      <circle cx="82" cy="40" r="24" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
      <text x="82" y="47" font-size="15" font-weight="800" text-anchor="middle" fill="#15803d">F_q</text>
      <text x="82" y="86" font-size="11.5" font-weight="800" text-anchor="middle" fill="#0f172a">Quantum Token</text>
      <text x="82" y="104" font-size="10.5" font-weight="800" text-anchor="middle" fill="#16a34a">F_q ∈ ℝ^(n_qubits)</text>
      <text x="82" y="122" font-size="9" text-anchor="middle" fill="#64748b">Hilbert Space Features</text>
      <rect x="15" y="132" width="135" height="46" rx="4" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1"/>
      <text x="82" y="150" font-size="9" font-weight="700" text-anchor="middle" fill="#15803d">Injected into Bottleneck</text>
      <text x="82" y="166" font-size="8.5" text-anchor="middle" fill="#475569">&amp; Severity MLP Fusion</text>
    </g>

    <!-- Connectors -->
    <path d="M 97 160 L 97 172" stroke="#16a34a" stroke-width="1.8" marker-end="url(#arrow)"/>
    <path d="M 97 260 L 97 272" stroke="#16a34a" stroke-width="1.8" marker-end="url(#arrow)"/>
    <path d="M 97 507 L 97 520" stroke="#16a34a" stroke-width="1.8" marker-end="url(#arrow)"/>
  </g>

  <!-- Transition Arrow: Quantum to Backbone -->
  <g transform="translate(760, 420)">
    <path d="M 0 0 L 45 0" stroke="#1e293b" stroke-width="2.5" marker-end="url(#arrow)"/>
    <text x="22" y="-12" font-size="9.5" font-weight="800" text-anchor="middle" fill="#0f172a">F_q Injection</text>
    <text x="22" y="18" font-size="8.5" font-weight="600" text-anchor="middle" fill="#64748b">8-Qubit State</text>
  </g>

  <!-- ==================== 4. MAIN SEGMENTATION BACKBONE ==================== -->
  <g id="panel-backbone" transform="translate(810, 80)">
    <rect width="530" height="740" rx="12" fill="url(#grad-backbone)" stroke="#d8b4fe" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="530" height="40" rx="12" fill="#e9d5ff"/>
    <rect x="0" y="20" width="530" height="20" fill="#e9d5ff"/>
    <!-- NO EMOJI! Clean typography -->
    <text x="265" y="19" font-size="14.5" font-weight="800" text-anchor="middle" fill="#581c87">MAIN SEGMENTATION BACKBONE</text>
    <text x="265" y="34" font-size="11.5" font-weight="600" text-anchor="middle" fill="#7e22ce">U-Net Style Dual-Branch Segmentation Network</text>

    <!-- ENCODER PATH (LEFT) -->
    <!-- Stage 1 Encoder -->
    <g transform="translate(16, 50)">
      <rect width="175" height="98" rx="8" fill="#ffffff" stroke="#9333ea" stroke-width="1.5" filter="url(#block-shadow)"/>
      <rect x="0" y="0" width="175" height="22" rx="8" fill="#f3e8ff"/>
      <rect x="0" y="12" width="175" height="10" fill="#f3e8ff"/>
      <text x="87" y="16" font-size="11" font-weight="800" text-anchor="middle" fill="#6b21a8">Stage 1 (Conv Modules)</text>
      <!-- 3D Slab Graphic -->
      <g transform="translate(8, 28)">
        <polygon points="5,8 15,2 35,2 25,8" fill="#93c5fd"/>
        <polygon points="25,8 35,2 35,48 25,54" fill="#3b82f6"/>
        <polygon points="5,8 25,8 25,54 5,54" fill="#60a5fa"/>
      </g>
      <text x="52" y="44" font-size="9.5" font-weight="700" text-anchor="start" fill="#0f172a">Conv-BN-ReLU Stem</text>
      <text x="52" y="58" font-size="9" text-anchor="start" fill="#475569">512×512 → 256×256</text>
      <text x="52" y="72" font-size="9.5" font-weight="800" text-anchor="start" fill="#7c3aed">C = 64 Channels</text>
      <text x="52" y="86" font-size="8.5" text-anchor="start" fill="#64748b">ResNet / SSM Stem</text>
    </g>

    <!-- Downsampling Arrow 1 -->
    <g transform="translate(103, 148)">
      <path d="M 0 0 L 0 32" stroke="#7e22ce" stroke-width="2" marker-end="url(#arrow-purple)"/>
      <text x="15" y="18" font-size="8.5" font-weight="700" fill="#7e22ce">Downsample (2× Stride)</text>
    </g>

    <!-- Stage 2 Encoder -->
    <g transform="translate(16, 182)">
      <rect width="175" height="102" rx="8" fill="#ffffff" stroke="#9333ea" stroke-width="1.5" filter="url(#block-shadow)"/>
      <rect x="0" y="0" width="175" height="22" rx="8" fill="#f3e8ff"/>
      <rect x="0" y="12" width="175" height="10" fill="#f3e8ff"/>
      <text x="87" y="16" font-size="11" font-weight="800" text-anchor="middle" fill="#6b21a8">Stage 2 (VSS Block 1)</text>
      <!-- 3D Slab Graphic -->
      <g transform="translate(8, 28)">
        <polygon points="5,8 15,2 35,2 25,8" fill="#d8b4fe"/>
        <polygon points="25,8 35,2 35,52 25,58" fill="#a855f7"/>
        <polygon points="5,8 25,8 25,58 5,58" fill="#c084fc"/>
      </g>
      <text x="52" y="44" font-size="9.5" font-weight="700" text-anchor="start" fill="#0f172a">VSS SS2D Block</text>
      <text x="52" y="58" font-size="9" text-anchor="start" fill="#475569">256×256 → 128×128</text>
      <text x="52" y="72" font-size="9.5" font-weight="800" text-anchor="start" fill="#7c3aed">C = 128 Channels</text>
      <!-- FFM Badge -->
      <circle cx="152" cy="65" r="11" fill="#dcfce7" stroke="#16a34a" stroke-width="1.2"/>
      <text x="152" y="69" font-size="8.5" font-weight="800" text-anchor="middle" fill="#15803d">FFM</text>
    </g>

    <!-- Downsampling Arrow 2 -->
    <g transform="translate(103, 284)">
      <path d="M 0 0 L 0 32" stroke="#7e22ce" stroke-width="2" marker-end="url(#arrow-purple)"/>
      <text x="15" y="18" font-size="8.5" font-weight="700" fill="#7e22ce">Patch Merging (2×)</text>
    </g>

    <!-- Stage 3 Encoder -->
    <g transform="translate(16, 318)">
      <rect width="175" height="102" rx="8" fill="#ffffff" stroke="#9333ea" stroke-width="1.5" filter="url(#block-shadow)"/>
      <rect x="0" y="0" width="175" height="22" rx="8" fill="#f3e8ff"/>
      <rect x="0" y="12" width="175" height="10" fill="#f3e8ff"/>
      <text x="87" y="16" font-size="11" font-weight="800" text-anchor="middle" fill="#6b21a8">Stage 3 (VSS Block 2)</text>
      <!-- 3D Slab Graphic -->
      <g transform="translate(8, 28)">
        <polygon points="5,8 15,2 35,2 25,8" fill="#fed7aa"/>
        <polygon points="25,8 35,2 35,52 25,58" fill="#f97316"/>
        <polygon points="5,8 25,8 25,58 5,58" fill="#fb923c"/>
      </g>
      <text x="52" y="44" font-size="9.5" font-weight="700" text-anchor="start" fill="#0f172a">VSS SS2D Block</text>
      <text x="52" y="58" font-size="9" text-anchor="start" fill="#475569">128×128 → 64×64</text>
      <text x="52" y="72" font-size="9.5" font-weight="800" text-anchor="start" fill="#7c3aed">C = 256 Channels</text>
      <!-- FFM Badge -->
      <circle cx="152" cy="65" r="11" fill="#dcfce7" stroke="#16a34a" stroke-width="1.2"/>
      <text x="152" y="69" font-size="8.5" font-weight="800" text-anchor="middle" fill="#15803d">FFM</text>
    </g>

    <!-- Downsampling Arrow 3 to Bottleneck -->
    <path d="M 103 420 L 103 460 L 175 490" stroke="#7e22ce" stroke-width="2" marker-end="url(#arrow-purple)" fill="none"/>
    <text x="90" y="450" font-size="8.5" font-weight="700" fill="#7e22ce">Downsample (2×)</text>

    <!-- BOTTLENECK (STAGE 4 / VAC) -->
    <g transform="translate(175, 450)">
      <rect width="180" height="105" rx="10" fill="#ffffff" stroke="#e11d48" stroke-width="2" filter="url(#block-shadow)"/>
      <rect x="0" y="0" width="180" height="22" rx="10" fill="#ffe4e6"/>
      <rect x="0" y="12" width="180" height="10" fill="#ffe4e6"/>
      <!-- NO EMOJI! Clean text -->
      <text x="90" y="16" font-size="11" font-weight="800" text-anchor="middle" fill="#9f1239">BOTTLENECK (STAGE 4)</text>
      <text x="90" y="38" font-size="10" font-weight="700" text-anchor="middle" fill="#0f172a">Quantum Attention Injection</text>
      <rect x="25" y="46" width="130" height="22" rx="4" fill="#fef2f2" stroke="#fecdd3" stroke-width="1"/>
      <text x="90" y="61" font-size="11" font-weight="800" text-anchor="middle" fill="#be123c">VAC MODULE</text>
      <text x="90" y="82" font-size="9.5" font-weight="800" text-anchor="middle" fill="#e11d48">32×32, C = 512</text>
      <text x="90" y="96" font-size="8.5" text-anchor="middle" fill="#64748b">Concat[F_bottleneck, F_q]</text>
    </g>

    <!-- Upsampling Arrow from Bottleneck to Decoder Stage 3' -->
    <path d="M 355 490 L 427 460 L 427 425" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)" fill="none"/>
    <text x="390" y="450" font-size="8.5" font-weight="700" fill="#2563eb">Upsample (2×)</text>

    <!-- DECODER PATH (RIGHT) -->
    <!-- Decoder Stage 3' -->
    <g transform="translate(340, 318)">
      <rect width="175" height="102" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="1.5" filter="url(#block-shadow)"/>
      <rect x="0" y="0" width="175" height="22" rx="8" fill="#dbeafe"/>
      <rect x="0" y="12" width="175" height="10" fill="#dbeafe"/>
      <text x="87" y="16" font-size="11" font-weight="800" text-anchor="middle" fill="#1e40af">Decoder Stage 3'</text>
      <!-- 3D Slab Graphic -->
      <g transform="translate(8, 28)">
        <polygon points="5,8 15,2 35,2 25,8" fill="#fed7aa"/>
        <polygon points="25,8 35,2 35,52 25,58" fill="#f97316"/>
        <polygon points="5,8 25,8 25,58 5,58" fill="#fb923c"/>
      </g>
      <text x="52" y="44" font-size="9.5" font-weight="700" text-anchor="start" fill="#0f172a">ConvTranspose2d</text>
      <text x="52" y="58" font-size="9" text-anchor="start" fill="#475569">32×32 → 64×64</text>
      <text x="52" y="72" font-size="9.5" font-weight="800" text-anchor="start" fill="#2563eb">C = 256 Channels</text>
      <!-- FFM Badge -->
      <circle cx="152" cy="65" r="11" fill="#dcfce7" stroke="#16a34a" stroke-width="1.2"/>
      <text x="152" y="69" font-size="8.5" font-weight="800" text-anchor="middle" fill="#15803d">FFM</text>
    </g>

    <!-- Upsampling Arrow 2 -->
    <g transform="translate(427, 318)">
      <path d="M 0 0 L 0 -30" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)"/>
      <text x="-12" y="-13" font-size="8.5" font-weight="700" fill="#2563eb" text-anchor="end">ConvTranspose (2×)</text>
    </g>

    <!-- Decoder Stage 2' -->
    <g transform="translate(340, 182)">
      <rect width="175" height="102" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="1.5" filter="url(#block-shadow)"/>
      <rect x="0" y="0" width="175" height="22" rx="8" fill="#dbeafe"/>
      <rect x="0" y="12" width="175" height="10" fill="#dbeafe"/>
      <text x="87" y="16" font-size="11" font-weight="800" text-anchor="middle" fill="#1e40af">Decoder Stage 2'</text>
      <!-- 3D Slab Graphic -->
      <g transform="translate(8, 28)">
        <polygon points="5,8 15,2 35,2 25,8" fill="#d8b4fe"/>
        <polygon points="25,8 35,2 35,52 25,58" fill="#a855f7"/>
        <polygon points="5,8 25,8 25,58 5,58" fill="#c084fc"/>
      </g>
      <text x="52" y="44" font-size="9.5" font-weight="700" text-anchor="start" fill="#0f172a">ConvTranspose2d</text>
      <text x="52" y="58" font-size="9" text-anchor="start" fill="#475569">64×64 → 128×128</text>
      <text x="52" y="72" font-size="9.5" font-weight="800" text-anchor="start" fill="#2563eb">C = 128 Channels</text>
      <!-- FFM Badge -->
      <circle cx="152" cy="65" r="11" fill="#dcfce7" stroke="#16a34a" stroke-width="1.2"/>
      <text x="152" y="69" font-size="8.5" font-weight="800" text-anchor="middle" fill="#15803d">FFM</text>
    </g>

    <!-- Upsampling Arrow 3 -->
    <g transform="translate(427, 182)">
      <path d="M 0 0 L 0 -30" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)"/>
      <text x="-12" y="-13" font-size="8.5" font-weight="700" fill="#2563eb" text-anchor="end">ConvTranspose (2×)</text>
    </g>

    <!-- Decoder Stage 1' -->
    <g transform="translate(340, 50)">
      <rect width="175" height="98" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="1.5" filter="url(#block-shadow)"/>
      <rect x="0" y="0" width="175" height="22" rx="8" fill="#dbeafe"/>
      <rect x="0" y="12" width="175" height="10" fill="#dbeafe"/>
      <text x="87" y="16" font-size="11" font-weight="800" text-anchor="middle" fill="#1e40af">Decoder Stage 1'</text>
      <!-- 3D Slab Graphic -->
      <g transform="translate(8, 28)">
        <polygon points="5,8 15,2 35,2 25,8" fill="#93c5fd"/>
        <polygon points="25,8 35,2 35,48 25,54" fill="#3b82f6"/>
        <polygon points="5,8 25,8 25,54 5,54" fill="#60a5fa"/>
      </g>
      <text x="52" y="44" font-size="9.5" font-weight="700" text-anchor="start" fill="#0f172a">Upsample + Conv</text>
      <text x="52" y="58" font-size="9" text-anchor="start" fill="#475569">128×128 → 512×512</text>
      <text x="52" y="72" font-size="9.5" font-weight="800" text-anchor="start" fill="#2563eb">C = 64 → 1</text>
      <text x="52" y="86" font-size="8.5" text-anchor="start" fill="#64748b">1×1 Conv → Sigmoid Head</text>
    </g>

    <!-- ================= SKIP CONNECTIONS ================= -->
    <!-- Skip 1 (Top) -->
    <path d="M 191 99 L 335 99" stroke="#64748b" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
    <rect x="210" y="87" width="105" height="24" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="262" y="103" font-size="9" font-weight="800" text-anchor="middle" fill="#334155">Skip 1: 512×512, C=64</text>

    <!-- Skip 2 (Middle) -->
    <path d="M 191 233 L 335 233" stroke="#64748b" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
    <rect x="205" y="221" width="115" height="24" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="262" y="237" font-size="9" font-weight="800" text-anchor="middle" fill="#334155">Skip 2: 128×128, C=128</text>

    <!-- Skip 3 (Lower) -->
    <path d="M 191 369 L 335 369" stroke="#64748b" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
    <rect x="205" y="357" width="115" height="24" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="262" y="373" font-size="9" font-weight="800" text-anchor="middle" fill="#334155">Skip 3: 64×64, C=256</text>

    <!-- Supervised Loss Arrow down -->
    <path d="M 265 560 L 265 595" stroke="#0284c7" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow-dashed)"/>
    <text x="265" y="582" font-size="9" font-weight="800" text-anchor="middle" fill="#0369a1" dx="55">Loss Supervision (L_seg)</text>

    <!-- Optimization Module Box -->
    <g transform="translate(55, 600)">
      <rect width="420" height="98" rx="8" fill="url(#grad-optim)" stroke="#38bdf8" stroke-width="1.8"/>
      <text x="210" y="22" font-size="12" font-weight="800" text-anchor="middle" fill="#0369a1">OPTIMIZATION MODULE: HYBRID MULTI-TASK LOSS</text>
      <text x="210" y="42" font-size="11" font-weight="800" text-anchor="middle" fill="#0f172a">L_total = α·L_seg + β·L_sev + γ·L_dam</text>
      <text x="210" y="60" font-size="9.5" text-anchor="middle" fill="#475569">L_seg = BCE + Dice + 3×Boundary_BCE  |  L_sev = CE + LabelSmooth(0.1)</text>
      <text x="210" y="78" font-size="9.5" font-weight="700" text-anchor="middle" fill="#0284c7">Bayesian Optimization (Optuna TPE) • Parameter-Shift Rule for VQC</text>
    </g>

    <!-- Feedback Loop Arrow to Models (Routes cleanly under all panels) -->
    <path d="M 55 650 L 12 650 L 12 755 L -425 755 L -425 615" stroke="#0284c7" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrow-dashed)" fill="none"/>
    <text x="-210" y="747" font-size="9.5" font-weight="800" text-anchor="middle" fill="#0284c7">Feedback (θ*) Parameter Updates via Adam &amp; Parameter-Shift Rule</text>
  </g>

  <!-- Backbone Output to Segmentation Mask Thumbnail -->
  <g transform="translate(1325, 180)">
    <path d="M 0 0 L 45 0" stroke="#1e293b" stroke-width="2.5" marker-end="url(#arrow)"/>
    <text x="22" y="-10" font-size="9" font-weight="800" text-anchor="middle" fill="#1e293b">Logits</text>
  </g>

  <!-- Segmentation Mask Card (Pure Vector) -->
  <g transform="translate(1375, 125)">
    <rect width="95" height="115" rx="8" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5" filter="url(#card-shadow)"/>
    <g transform="translate(12, 10)">
      <rect width="71" height="68" rx="4" fill="#1e1b4b"/>
      <!-- Multi-tone crack heatmap -->
      <path d="M 35 5 Q 32 22 38 38 T 44 63" stroke="#a855f7" stroke-width="6" fill="none" opacity="0.6"/>
      <path d="M 38 38 Q 50 48 58 56" stroke="#a855f7" stroke-width="4.5" fill="none" opacity="0.6"/>
      <path d="M 35 24 Q 22 32 15 42" stroke="#a855f7" stroke-width="4.5" fill="none" opacity="0.6"/>
      <!-- Core crack line -->
      <path d="M 35 5 Q 32 22 38 38 T 44 63" stroke="#e9d5ff" stroke-width="2" fill="none"/>
      <path d="M 38 38 Q 50 48 58 56" stroke="#e9d5ff" stroke-width="1.6" fill="none"/>
      <path d="M 35 24 Q 22 32 15 42" stroke="#e9d5ff" stroke-width="1.6" fill="none"/>
    </g>
    <text x="47" y="93" font-size="10" font-weight="800" text-anchor="middle" fill="#0f172a">Segmentation</text>
    <text x="47" y="106" font-size="9.5" font-weight="700" text-anchor="middle" fill="#7c3aed">Mask (M_seg)</text>
  </g>

  <!-- Arrow into Refinement -->
  <g transform="translate(1470, 180)">
    <path d="M 0 0 L 40 0" stroke="#1e293b" stroke-width="2.5" marker-end="url(#arrow)"/>
  </g>

  <!-- ==================== 5. QUANTUM-ENHANCED BOUNDARY REFINEMENT MODULE ==================== -->
  <g id="panel-refinement" transform="translate(1515, 80)">
    <rect width="185" height="315" rx="12" fill="url(#grad-refine)" stroke="#fda4af" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="185" height="40" rx="12" fill="#fecdd3"/>
    <rect x="0" y="20" width="185" height="20" fill="#fecdd3"/>
    <!-- NO EMOJI! Clean professional text -->
    <text x="92" y="18" font-size="12" font-weight="800" text-anchor="middle" fill="#9f1239">QUANTUM-ENHANCED</text>
    <text x="92" y="33" font-size="11.5" font-weight="800" text-anchor="middle" fill="#9f1239">BOUNDARY REFINEMENT</text>

    <!-- Sub-block 1: Quantum Attention -->
    <g transform="translate(15, 52)">
      <rect width="155" height="78" rx="6" fill="#ffffff" stroke="#fb7185" stroke-width="1.2"/>
      <text x="77" y="22" font-size="11.5" font-weight="800" text-anchor="middle" fill="#be123c">Quantum Attention</text>
      <text x="77" y="38" font-size="9.5" text-anchor="middle" fill="#475569">Kernel State Fidelity</text>
      <text x="77" y="52" font-size="9" font-weight="700" text-anchor="middle" fill="#e11d48">K(p_i, p_j) = |⟨ψ_i|ψ_j⟩|²</text>
      <text x="77" y="68" font-size="8.5" text-anchor="middle" fill="#64748b">Hilbert Boundary Focus</text>
    </g>

    <!-- Arrow down -->
    <path d="M 92 132 L 92 160" stroke="#e11d48" stroke-width="2" marker-end="url(#arrow-rose)"/>
    <text x="130" y="148" font-size="8.5" font-weight="700" fill="#e11d48">Kernel Guidance</text>

    <!-- Sub-block 2: Refinement Head -->
    <g transform="translate(15, 168)">
      <rect width="155" height="115" rx="6" fill="#ffffff" stroke="#fb7185" stroke-width="1.2"/>
      <text x="77" y="22" font-size="11.5" font-weight="800" text-anchor="middle" fill="#be123c">Refinement Head</text>
      <text x="77" y="40" font-size="9.5" text-anchor="middle" fill="#475569">Dense CRF Integration</text>
      <text x="77" y="56" font-size="9" text-anchor="middle" fill="#475569">Boundary Weighted BCE</text>
      <text x="77" y="72" font-size="9" text-anchor="middle" fill="#475569">Morphological Cleanup</text>
      <text x="77" y="90" font-size="8.5" font-weight="700" text-anchor="middle" fill="#be123c">Removes &lt;50px Islands</text>
      <text x="77" y="104" font-size="8.5" text-anchor="middle" fill="#64748b">Single-pixel Hole Filling</text>
    </g>
  </g>

  <!-- Transition Arrow: Refinement to Binary Crack Mask -->
  <g transform="translate(1700, 180)">
    <path d="M 0 0 L 40 0" stroke="#1e293b" stroke-width="2.5" marker-end="url(#arrow)"/>
  </g>

  <!-- Binary Crack Mask Card (Pure Vector) -->
  <g transform="translate(1745, 125)">
    <rect width="95" height="115" rx="8" fill="#ffffff" stroke="#94a3b8" stroke-width="1.5" filter="url(#card-shadow)"/>
    <g transform="translate(12, 10)">
      <rect width="71" height="68" rx="4" fill="#020617"/>
      <!-- Fine white skeleton crack lines -->
      <path d="M 35 5 Q 32 22 38 38 T 44 63" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" fill="none"/>
      <path d="M 38 38 Q 50 48 58 56" stroke="#ffffff" stroke-width="2" stroke-linecap="round" fill="none"/>
      <path d="M 35 24 Q 22 32 15 42" stroke="#ffffff" stroke-width="2" stroke-linecap="round" fill="none"/>
    </g>
    <text x="47" y="93" font-size="10" font-weight="800" text-anchor="middle" fill="#0f172a">Binary Crack</text>
    <text x="47" y="106" font-size="9.5" font-weight="700" text-anchor="middle" fill="#1e293b">Mask (M_crack)</text>
  </g>

  <!-- Transition Arrow: Binary Mask to Final Outputs -->
  <g transform="translate(1840, 180)">
    <path d="M 0 0 L 35 0" stroke="#1e293b" stroke-width="2.5" marker-end="url(#arrow)"/>
  </g>

  <!-- Path from Binary Mask down to Structural & Severity Modules -->
  <path d="M 1792 240 L 1792 415 L 1445 415 L 1445 435" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)" fill="none"/>
  <text x="1610" y="407" font-size="9.5" font-weight="800" text-anchor="middle" fill="#1e293b">Downstream Flow: Mask to Structural &amp; Severity Modules</text>

  <!-- ==================== 6. STRUCTURAL DESCRIPTOR LEARNING ==================== -->
  <g id="panel-structural" transform="translate(1365, 435)">
    <rect width="155" height="385" rx="12" fill="url(#grad-struct)" stroke="#fcd34d" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="155" height="40" rx="12" fill="#fde68a"/>
    <rect x="0" y="20" width="155" height="20" fill="#fde68a"/>
    <text x="77" y="18" font-size="12" font-weight="800" text-anchor="middle" fill="#78350f">STRUCTURAL</text>
    <text x="77" y="32" font-size="11.5" font-weight="800" text-anchor="middle" fill="#78350f">DESCRIPTORS</text>

    <!-- Structural Graphic -->
    <g transform="translate(25, 48)">
      <image href="{structural_b64}" x="15" y="0" width="75" height="72" preserveAspectRatio="xMidYMid contain"/>
    </g>

    <!-- Sub-block 1: Geometry -->
    <g transform="translate(12, 126)">
      <rect width="131" height="70" rx="6" fill="#ffffff" stroke="#f59e0b" stroke-width="1"/>
      <text x="65" y="17" font-size="10" font-weight="800" text-anchor="middle" fill="#b45309">Crack Geometry</text>
      <text x="65" y="32" font-size="8.5" text-anchor="middle" fill="#475569">Zhang-Suen Thinning</text>
      <text x="65" y="46" font-size="8.5" text-anchor="middle" fill="#475569">Euclidean Dist Trans</text>
      <text x="65" y="60" font-size="8.5" font-weight="700" text-anchor="middle" fill="#78350f">Length, Width, Area</text>
    </g>

    <!-- Sub-block 2: Topology -->
    <g transform="translate(12, 204)">
      <rect width="131" height="70" rx="6" fill="#ffffff" stroke="#f59e0b" stroke-width="1"/>
      <text x="65" y="17" font-size="10" font-weight="800" text-anchor="middle" fill="#b45309">Distress Topology</text>
      <text x="65" y="32" font-size="8.5" text-anchor="middle" fill="#475569">Density ρ = A_c / A_t</text>
      <text x="65" y="46" font-size="8.5" text-anchor="middle" fill="#475569">8-Connected Comps</text>
      <text x="65" y="60" font-size="8.5" font-weight="700" text-anchor="middle" fill="#78350f">Branching Junctions</text>
    </g>

    <!-- Sub-block 3: Network -->
    <g transform="translate(12, 282)">
      <rect width="131" height="85" rx="6" fill="#ffffff" stroke="#f59e0b" stroke-width="1"/>
      <text x="65" y="17" font-size="10" font-weight="800" text-anchor="middle" fill="#b45309">Network Graph</text>
      <text x="65" y="32" font-size="8.5" text-anchor="middle" fill="#475569">Nodes: Junctions</text>
      <text x="65" y="46" font-size="8.5" text-anchor="middle" fill="#475569">Edges: Crack Segments</text>
      <text x="65" y="60" font-size="8.5" text-anchor="middle" fill="#64748b">18-Bin Orientation Hist</text>
      <text x="65" y="75" font-size="9" font-weight="800" text-anchor="middle" fill="#b45309">Vector F_g ∈ ℝ^d</text>
    </g>
  </g>

  <!-- Arrow from Structural to Severity -->
  <g transform="translate(1520, 630)">
    <path d="M 0 0 L 35 0" stroke="#1e293b" stroke-width="2.5" marker-end="url(#arrow)"/>
    <text x="17" y="-10" font-size="9" font-weight="800" text-anchor="middle" fill="#78350f">F_g</text>
  </g>

  <!-- ==================== 7. SEVERITY / PCI / DETERIORATION ==================== -->
  <g id="panel-severity" transform="translate(1560, 435)">
    <rect width="210" height="385" rx="12" fill="url(#grad-severity)" stroke="#fdba74" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="210" height="40" rx="12" fill="#fed7aa"/>
    <rect x="0" y="20" width="210" height="20" fill="#fed7aa"/>
    <!-- NO EMOJI! Clean professional text -->
    <text x="105" y="18" font-size="12" font-weight="800" text-anchor="middle" fill="#9a3412">SEVERITY, PCI &amp;</text>
    <text x="105" y="32" font-size="11.5" font-weight="800" text-anchor="middle" fill="#9a3412">DETERIORATION MODELING</text>

    <!-- Sub-block 1: Severity Prediction (Hybrid Quantum NN) -->
    <g transform="translate(14, 48)">
      <rect width="182" height="120" rx="6" fill="#ffffff" stroke="#f97316" stroke-width="1.5"/>
      <!-- NO EMOJI! -->
      <text x="91" y="18" font-size="11" font-weight="800" text-anchor="middle" fill="#c2410c">Severity Prediction</text>
      <text x="91" y="33" font-size="9.5" font-weight="700" text-anchor="middle" fill="#ea580c">Hybrid Quantum NN</text>
      <line x1="8" y1="39" x2="174" y2="39" stroke="#fed7aa" stroke-width="1"/>
      <text x="91" y="53" font-size="8.5" text-anchor="middle" fill="#475569">Fusion: [F_curv,F_tex,F_sem,F_g,F_q]</text>
      <text x="91" y="67" font-size="8.5" text-anchor="middle" fill="#475569">Head: FC(d→128→64→8) → Tanh</text>
      <text x="91" y="82" font-size="9.5" font-weight="700" text-anchor="middle" fill="#c2410c">6-Layer VQC + Pauli-Z</text>
      <text x="91" y="96" font-size="8.5" text-anchor="middle" fill="#475569">Tail: FC(8→32→4) → Softmax</text>
      <rect x="25" y="100" width="132" height="15" rx="3" fill="#ffedd5"/>
      <text x="91" y="112" font-size="8.5" font-weight="800" text-anchor="middle" fill="#9a3412">Outputs: Severity &amp; DI</text>
    </g>

    <!-- Arrow down -->
    <path d="M 105 170 L 105 184" stroke="#f97316" stroke-width="1.8" marker-end="url(#arrow-orange)"/>

    <!-- Sub-block 2: PCI Estimation -->
    <g transform="translate(14, 186)">
      <rect width="182" height="96" rx="6" fill="#ffffff" stroke="#f97316" stroke-width="1.2"/>
      <text x="91" y="18" font-size="11" font-weight="800" text-anchor="middle" fill="#c2410c">PCI Estimation</text>
      <!-- Gauge Graphic -->
      <path d="M 66 45 A 25 25 0 0 1 116 45" fill="none" stroke="#22c55e" stroke-width="5"/>
      <path d="M 91 45 A 25 25 0 0 1 116 45" fill="none" stroke="#eab308" stroke-width="5"/>
      <path d="M 107 45 A 25 25 0 0 1 116 45" fill="none" stroke="#ef4444" stroke-width="5"/>
      <line x1="91" y1="45" x2="100" y2="29" stroke="#0f172a" stroke-width="2"/>
      <text x="91" y="63" font-size="9" text-anchor="middle" fill="#475569">ASTM D6433 Standard</text>
      <text x="91" y="76" font-size="9" text-anchor="middle" fill="#475569">TDV = Σ Deducts | CDV Curve</text>
      <text x="91" y="90" font-size="9.5" font-weight="800" text-anchor="middle" fill="#c2410c">PCI = 100 - max(CDV)</text>
    </g>

    <!-- Arrow down -->
    <path d="M 105 284 L 105 298" stroke="#f97316" stroke-width="1.8" marker-end="url(#arrow-orange)"/>

    <!-- Sub-block 3: Deterioration Modeling -->
    <g transform="translate(14, 300)">
      <rect width="182" height="74" rx="6" fill="#ffffff" stroke="#f97316" stroke-width="1.2"/>
      <text x="91" y="18" font-size="11" font-weight="800" text-anchor="middle" fill="#c2410c">Deterioration Modeling</text>
      <path d="M 35 48 Q 85 45 110 32 T 155 22" fill="none" stroke="#f97316" stroke-width="2.2"/>
      <text x="91" y="56" font-size="9" text-anchor="middle" fill="#475569">Markov Chain / Sigmoidal</text>
      <text x="91" y="69" font-size="8.5" font-weight="700" text-anchor="middle" fill="#9a3412">Maintenance Urgency Index</text>
    </g>
  </g>

  <!-- Arrows from Severity Module to Final Outputs -->
  <g transform="translate(1770, 495)">
    <path d="M 0 0 L 105 0" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>
    <text x="52" y="-10" font-size="9.5" font-weight="800" text-anchor="middle" fill="#c2410c">DI &amp; Severity</text>
  </g>

  <g transform="translate(1770, 650)">
    <path d="M 0 0 L 105 0" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>
    <text x="52" y="-10" font-size="9.5" font-weight="800" text-anchor="middle" fill="#c2410c">PCI Metric</text>
  </g>

  <!-- ==================== 8. FINAL OUTPUTS ==================== -->
  <g id="panel-outputs" transform="translate(1875, 80)">
    <rect width="185" height="740" rx="12" fill="url(#grad-outputs)" stroke="#94a3b8" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="185" height="40" rx="12" fill="#cbd5e1"/>
    <rect x="0" y="20" width="185" height="20" fill="#cbd5e1"/>
    <text x="92" y="26" font-size="13.5" font-weight="800" text-anchor="middle" fill="#0f172a">FINAL OUTPUTS</text>

    <!-- Output 1: Refined Crack Mask -->
    <g transform="translate(17, 50)">
      <rect width="150" height="152" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>
      <g transform="translate(40, 10)">
        <rect width="70" height="68" rx="4" fill="#020617"/>
        <path d="M 35 5 Q 32 22 38 38 T 44 63" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" fill="none"/>
        <path d="M 38 38 Q 50 48 58 56" stroke="#ffffff" stroke-width="2" stroke-linecap="round" fill="none"/>
        <path d="M 35 24 Q 22 32 15 42" stroke="#ffffff" stroke-width="2" stroke-linecap="round" fill="none"/>
      </g>
      <text x="75" y="104" font-size="10.5" font-weight="800" text-anchor="middle" fill="#0f172a">Refined Binary Crack</text>
      <text x="75" y="120" font-size="10" font-weight="800" text-anchor="middle" fill="#2563eb">Segmentation Mask</text>
      <text x="75" y="136" font-size="9" text-anchor="middle" fill="#64748b">Pixel-level Boundary</text>
    </g>

    <!-- Output 2: Severity Classification -->
    <g transform="translate(17, 220)">
      <rect width="150" height="152" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>
      <!-- Traffic Light Circles -->
      <circle cx="45" cy="30" r="13" fill="#22c55e"/>
      <circle cx="75" cy="30" r="13" fill="#eab308"/>
      <circle cx="105" cy="30" r="13" fill="#ef4444"/>
      <text x="75" y="62" font-size="11.5" font-weight="800" text-anchor="middle" fill="#0f172a">Severity Rating</text>
      <text x="75" y="80" font-size="9" text-anchor="middle" fill="#475569">• Low (0): &lt;1mm Hairline</text>
      <text x="75" y="94" font-size="9" text-anchor="middle" fill="#475569">• Moderate (1): 1–3mm</text>
      <text x="75" y="108" font-size="9" text-anchor="middle" fill="#475569">• Severe (2): 3–10mm</text>
      <text x="75" y="122" font-size="9" text-anchor="middle" fill="#475569">• Critical (3): &gt;10mm Netz</text>
      <text x="75" y="138" font-size="9.5" font-weight="800" text-anchor="middle" fill="#ea580c">4-Level ASTM Label</text>
    </g>

    <!-- Output 3: Pavement Damage Index (PDI) -->
    <g transform="translate(17, 390)">
      <rect width="150" height="152" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>
      <!-- Gauge -->
      <path d="M 40 60 A 35 35 0 0 1 110 60" fill="none" stroke="#22c55e" stroke-width="8"/>
      <path d="M 75 60 A 35 35 0 0 1 110 60" fill="none" stroke="#eab308" stroke-width="8"/>
      <path d="M 95 60 A 35 35 0 0 1 110 60" fill="none" stroke="#ef4444" stroke-width="8"/>
      <line x1="75" y1="60" x2="93" y2="38" stroke="#0f172a" stroke-width="2.5"/>
      <circle cx="75" cy="60" r="4" fill="#0f172a"/>
      <text x="75" y="86" font-size="11.5" font-weight="800" text-anchor="middle" fill="#0f172a">Damage Index (DI)</text>
      <text x="75" y="102" font-size="9.5" text-anchor="middle" fill="#475569">Pavement Distress (PDI)</text>
      <text x="75" y="118" font-size="9" text-anchor="middle" fill="#64748b">Composite Extent Score</text>
      <text x="75" y="136" font-size="10.5" font-weight="800" text-anchor="middle" fill="#c2410c">Score: 0.68 / 1.0</text>
    </g>

    <!-- Output 4: Pavement Condition Index (PCI) -->
    <g transform="translate(17, 560)">
      <rect width="150" height="152" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>
      <!-- Bar Chart -->
      <rect x="38" y="42" width="18" height="35" fill="#ef4444" rx="2"/>
      <rect x="63" y="29" width="18" height="48" fill="#eab308" rx="2"/>
      <rect x="88" y="15" width="18" height="62" fill="#22c55e" rx="2"/>
      <line x1="28" y1="77" x2="122" y2="77" stroke="#94a3b8" stroke-width="1.5"/>
      <text x="75" y="98" font-size="11.5" font-weight="800" text-anchor="middle" fill="#0f172a">PCI Assessment</text>
      <text x="75" y="114" font-size="12" font-weight="800" text-anchor="middle" fill="#16a34a">Score: 78 / 100</text>
      <text x="75" y="130" font-size="9" text-anchor="middle" fill="#475569">Condition: Satisfactory</text>
      <text x="75" y="142" font-size="8.5" text-anchor="middle" fill="#64748b">ASTM D6433 Survey</text>
    </g>
  </g>

  <!-- ==================== 9. LEGEND (EXACTLY MATCHING REFERENCE IMAGE!) ==================== -->
  <g id="panel-legend" transform="translate(2080, 80)">
    <rect width="320" height="740" rx="12" fill="url(#grad-legend)" stroke="#64748b" stroke-width="1.5" filter="url(#card-shadow)"/>
    <rect x="0" y="0" width="320" height="40" rx="12" fill="#f1f5f9"/>
    <rect x="0" y="20" width="320" height="20" fill="#f1f5f9"/>
    <text x="160" y="26" font-size="14" font-weight="800" text-anchor="middle" fill="#0f172a">LEGEND &amp; NOTATION</text>

    <!-- SECTION 1: ARROW STYLES -->
    <g transform="translate(15, 52)">
      <text x="0" y="14" font-size="11.5" font-weight="800" fill="#0f172a">1. Transition &amp; Arrow Types</text>
      
      <!-- Input / Output Arrow -->
      <g transform="translate(0, 26)">
        <path d="M 5 8 L 55 8" stroke="#1e293b" stroke-width="3" marker-end="url(#arrow)"/>
        <text x="70" y="12" font-size="10.5" font-weight="700" fill="#1e293b">Input / Output Data Flow</text>
      </g>

      <!-- Skip Connection Arrow -->
      <g transform="translate(0, 52)">
        <path d="M 5 8 L 55 8" stroke="#64748b" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrow)"/>
        <text x="70" y="12" font-size="10.5" font-weight="700" fill="#334155">Skip Connection (U-Net)</text>
      </g>

      <!-- Downsampling Transition Arrow -->
      <g transform="translate(0, 78)">
        <path d="M 30 2 L 30 20" stroke="#7e22ce" stroke-width="2.5" marker-end="url(#arrow-purple)"/>
        <text x="70" y="14" font-size="10.5" font-weight="700" fill="#6b21a8">Downsampling (Stride 2)</text>
      </g>

      <!-- Upsampling Transition Arrow -->
      <g transform="translate(0, 104)">
        <path d="M 30 20 L 30 2" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
        <text x="70" y="14" font-size="10.5" font-weight="700" fill="#1e40af">Upsampling (ConvTranspose)</text>
      </g>

      <!-- Loss / Feedback Loop Arrow -->
      <g transform="translate(0, 130)">
        <path d="M 5 8 L 55 8" stroke="#0284c7" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrow-dashed)"/>
        <text x="70" y="12" font-size="10.5" font-weight="700" fill="#0369a1">Feedback / Supervision (θ*)</text>
      </g>
    </g>

    <line x1="15" y1="210" x2="305" y2="210" stroke="#e2e8f0" stroke-width="1.5"/>

    <!-- SECTION 2: OPERATOR SYMBOLS -->
    <g transform="translate(15, 222)">
      <text x="0" y="14" font-size="11.5" font-weight="800" fill="#0f172a">2. Mathematical Operators &amp; Blocks</text>

      <!-- Addition -->
      <g transform="translate(10, 24)">
        <circle cx="15" cy="8" r="10" fill="#f8fafc" stroke="#0f172a" stroke-width="1.5"/>
        <text x="15" y="12" font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">⊕</text>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e293b">Element-wise Addition</text>
      </g>

      <!-- Channel Concatenation -->
      <g transform="translate(10, 50)">
        <circle cx="15" cy="8" r="10" fill="#f8fafc" stroke="#0f172a" stroke-width="1.5"/>
        <text x="15" y="12" font-size="12" font-weight="800" text-anchor="middle" fill="#0f172a">C</text>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e293b">Channel Concatenation</text>
      </g>

      <!-- Matrix Multiplication / Attention -->
      <g transform="translate(10, 76)">
        <circle cx="15" cy="8" r="10" fill="#f8fafc" stroke="#0f172a" stroke-width="1.5"/>
        <text x="15" y="12" font-size="12" font-weight="700" text-anchor="middle" fill="#0f172a">⊗</text>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e293b">Matrix Multiplication</text>
      </g>

      <!-- VSS Block -->
      <g transform="translate(10, 102)">
        <rect x="3" y="-1" width="24" height="18" rx="4" fill="#f3e8ff" stroke="#9333ea" stroke-width="1.2"/>
        <text x="15" y="12" font-size="9" font-weight="800" text-anchor="middle" fill="#6b21a8">VSS</text>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e293b">Visual State Space (SS2D)</text>
      </g>

      <!-- FFM Module -->
      <g transform="translate(10, 128)">
        <circle cx="15" cy="8" r="11" fill="#dcfce7" stroke="#16a34a" stroke-width="1.2"/>
        <text x="15" y="12" font-size="8.5" font-weight="800" text-anchor="middle" fill="#15803d">FFM</text>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e293b">Feature Fusion Module</text>
      </g>

      <!-- VAC Module (NO EMOJI) -->
      <g transform="translate(10, 154)">
        <rect x="2" y="-1" width="26" height="18" rx="4" fill="#ffe4e6" stroke="#e11d48" stroke-width="1.2"/>
        <text x="15" y="12" font-size="9" font-weight="800" text-anchor="middle" fill="#be123c">VAC</text>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e293b">Visual Attention Circuit</text>
      </g>
    </g>

    <line x1="15" y1="410" x2="305" y2="410" stroke="#e2e8f0" stroke-width="1.5"/>

    <!-- SECTION 3: FEATURE DOMAINS & COLOR CODES (MATCHING REFERENCE) -->
    <g transform="translate(15, 425)">
      <text x="0" y="14" font-size="11.5" font-weight="800" fill="#0f172a">3. Feature Domain Notation</text>

      <!-- Curvilinear Feature -->
      <g transform="translate(10, 24)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#93c5fd" stroke="#3b82f6" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e40af">Curvilinear Ridge (F_curv)</text>
      </g>

      <!-- Spectral Feature -->
      <g transform="translate(10, 48)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#bfdbfe" stroke="#2563eb" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e40af">Spectral / Gabor (F_tex)</text>
      </g>

      <!-- Semantic Feature -->
      <g transform="translate(10, 72)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#d8b4fe" stroke="#a855f7" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#6b21a8">Mamba Semantic (F_sem)</text>
      </g>

      <!-- Quantum Feature -->
      <g transform="translate(10, 96)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#bbf7d0" stroke="#22c55e" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#15803d">Quantum VQC (F_q)</text>
      </g>

      <!-- Fused Feature -->
      <g transform="translate(10, 120)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#fed7aa" stroke="#f97316" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#9a3412">Fused Vector (F_c / F_sev)</text>
      </g>

      <!-- Structural Feature -->
      <g transform="translate(10, 144)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#fde68a" stroke="#eab308" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#854d0e">Structural Descriptors (F_g)</text>
      </g>

      <!-- Spatial Domain Feature Block (Matching reference SF) -->
      <g transform="translate(10, 168)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#bbf7d0" stroke="#16a34a" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#14532d">Spatial Domain Feature (SF)</text>
      </g>

      <!-- Low Frequency Feature Block (Matching reference LF) -->
      <g transform="translate(10, 192)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#93c5fd" stroke="#2563eb" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#1e40af">Low Frequency Feature (LF)</text>
      </g>

      <!-- High Frequency Feature Block (Matching reference HF) -->
      <g transform="translate(10, 216)">
        <rect x="5" y="0" width="20" height="16" rx="3" fill="#fed7aa" stroke="#ea580c" stroke-width="1"/>
        <text x="45" y="12" font-size="10.5" font-weight="600" fill="#9a3412">High Frequency Feature (HF)</text>
      </g>
    </g>
  </g>

  <!-- ==================== FOOTER / SUB-CAPTION (IEEE STYLE) ==================== -->
  <g id="figure-caption" transform="translate(1210, 930)">
    <text x="0" y="0" font-size="16" font-weight="800" text-anchor="middle" fill="#0f172a">(a) Overview of the Quantum-Enhanced Pavement Condition Monitoring System (Q-PCMS)</text>
    <text x="0" y="22" font-size="12.5" font-weight="500" text-anchor="middle" fill="#64748b">Detailed layer-wise dual-branch U-Net backbone with Visual State Space (VSS) blocks, Feature Fusion Modules (FFM), Variational Quantum Attention Circuit (VAC) bottleneck, and Hybrid Quantum Severity Classifier.</text>
  </g>
</svg>
"""
    return svg

def main():
    svg_content = generate_svg()
    
    out_svg = Path(r"c:\Users\Arunabh\Desktop\PROJ1\architecture_diagram.svg")
    out_svg.write_text(svg_content, encoding="utf-8")
    print(f"Vector SVG saved to: {out_svg}")

    out_html = Path(r"c:\Users\Arunabh\Desktop\PROJ1\architecture_diagram.html")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Architecture Diagram — Quantum-Enhanced Pavement Condition Monitoring System</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background-color: #ffffff;
      display: flex;
      justify-content: center;
      align-items: center;
    }}
    svg {{
      width: 2420px;
      height: 1040px;
    }}
  </style>
</head>
<body>
{svg_content}
</body>
</html>"""
    out_html.write_text(html_content, encoding="utf-8")
    print(f"HTML viewer saved to: {out_html}")

    out_png = Path(r"c:\Users\Arunabh\Desktop\PROJ1\architecture_diagram.png")
    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    
    if os.path.exists(edge_exe):
        cmd = [
            edge_exe,
            "--headless=new",
            f"--screenshot={out_png.resolve()}",
            "--window-size=2420,1050",
            "--hide-scrollbars",
            out_html.resolve().as_uri()
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if out_png.exists():
            print(f"High-res PNG rendered to: {out_png} (Size: {out_png.stat().st_size} bytes)")
        else:
            print("Edge render failed:", res.stderr)
    else:
        print("Edge executable not found, SVG is ready.")

if __name__ == "__main__":
    main()
