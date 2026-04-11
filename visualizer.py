import plotly.graph_objects as go
import numpy as np

def plot_multiscale():
    x = np.loadtxt('output/x.txt')
    u = np.loadtxt('output/displacement.txt')
    stress = np.loadtxt('output/stress.txt')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=u*1e6, name='Displacement (μm)', 
                            line=dict(color='blue', width=4)))
    fig.add_trace(go.Scatter(x=x, y=stress/1e6, name='Stress (MPa)', 
                            line=dict(color='red', width=4), yaxis='y2'))
    
    fig.update_layout(
        title="Multi-Scale Beam Analysis: Displacement & Stress",
        xaxis_title="Position (m)",
        yaxis_title="Displacement (μm)",
        yaxis2=dict(title="Stress (MPa)", overlaying='y', side='right'),
        height=500,
        template='plotly_white'
    )
    
    fig.write_html('docs/index.html')
    fig.show()
    print("Interactive plot in browser + docs/index.html saved!")
    return fig

if __name__ == "__main__":
    plot_multiscale()