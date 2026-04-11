import streamlit as st
import numpy as np
from src.solver import solve_elastostatics
from src.visualizer import plot_multiscale

st.set_page_config(page_title="Elastostatics Visualizer", layout="wide")
st.title("🧮 Multi-Scale Elastostatics Visualizer")
st.markdown("**Interactive FEM solver for mechanical engineering portfolios.** Adjust parameters → Solve live!")

col1, col2 = st.columns(2)
with col1:
    st.header("Material Properties")
    E_macro = st.slider("Macro Modulus (GPa)", 10.0, 200.0, 50.0) * 1e9
    E_micro = st.slider("Micro Modulus (GPa)", 1.0, 50.0, 10.0) * 1e9
with col2:
    st.header("Loading")
    load = st.slider("Axial Load (N)", 1e5, 5e6, 1e6)
    length = st.slider("Beam Length (m)", 0.5, 2.0, 1.0)

if st.button("🚀 Solve Multi-Scale Problem", type="primary", use_container_width=True):
    with st.spinner("Solving FE equations..."):
        results = solve_elastostatics(E_macro, E_micro, length, load)
    
    st.success(f"E_effective = {results['E_eff']/1e9:.1f} GPa | Max disp = {np.max(np.abs(results['u']))*1e6:.1f} μm")
    
    fig = plot_multiscale()
    st.plotly_chart(fig, use_container_width=True)
    
    with open('output/displacement.txt', 'r') as f:
        st.download_button("📥 Download Displacement", f.read(), "displacement.txt")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        with open('output/stress.txt', 'r') as f:
            st.download_button("📊 Stress", f.read(), "stress.txt")
    with col_dl2:
        with open('output/x.txt', 'r') as f:
            st.download_button("📐 Coordinates", f.read(), "x.txt")

st.markdown("---")
