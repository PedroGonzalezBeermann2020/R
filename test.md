fig = go.Figure(data=[go.Surface(z=z, x=x_grid, y=y_grid, 
    contours = {
        "x": {"show": True, "start": 0, "end": 1.0, "size": 0.05, "color":"black"},
        "y": {"show": True, "start": 0, "end": 1.0, "size": 0.05, "color":"black"},
        "z": {"show": True, "start": 87.0, "end": 100, "size": 0.2}
    },)])
fig.update_layout(title='Superficie de Respuesta', autosize=True,width=800, height=800)
fig.update_traces(contours_z=dict(show=True, usecolormap=True,highlightcolor="limegreen", project_z=True))
fig.show()
