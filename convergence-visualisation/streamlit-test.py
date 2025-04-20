import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from src.entities import Quadcopter, PointOfInterest, AreaOfInterest
from src.voronoi_utils import clipped_voronoi_polygons_2d
from src.planner import assign_voronoi_targets
from src.visualization import draw_static, animate_simulation

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import io
pio.renderers.default = "browser"  # Set default renderer to browser



# Map dimensions
MAP_WIDTH = 160.0
MAP_HEIGHT = 90.0

# --- App Title ---
st.title("Drone Map Setup")

# --- Sidebar Controls ---
num_drones = st.sidebar.slider(
    "Number of drones",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)
death_min, death_max = st.sidebar.slider(
    "Drone death probability range",
    min_value=0.0,
    max_value=0.20,
    value=(0.0, 0.1),
    step=0.005
)

# --- POI Controls ---
num_pois = st.sidebar.slider(
    "Number of POIs",
    min_value=1,
    max_value=50,
    value=10,
    step=1
)
poi_min, poi_max = st.sidebar.slider(
    "POI weight range",
    min_value=0.5,
    max_value=20.0,
    value=(1.0, 10.0),
    step=0.5
)

# --- AOI Controls ---
num_aois = st.sidebar.slider(
    "Number of Areas of Interest",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)
aoi_min_weight, aoi_max_weight = st.sidebar.slider(
    "AOI weight range",
    min_value=1.0,
    max_value=10.0,
    value=(1.0, 5.0),
    step=0.5
)

if st.sidebar.button("Generate random AOIs", key="generate_random_aois"):
    # Generate random AOIs
    st.session_state.aois = []
    for _ in range(num_aois):
        # Generate random center position for the AOI
        cx = float(np.random.uniform(0, MAP_WIDTH))
        cy = float(np.random.uniform(0, MAP_HEIGHT))
        
        # Generate random number of vertices and angles
        num_vertices = np.random.randint(3, 9)  # AOIs must have at least 3 vertices
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
        
        # Generate random radii and calculate polygon coordinates
        r_min, r_max = min(MAP_WIDTH, MAP_HEIGHT) * 0.05, min(MAP_WIDTH, MAP_HEIGHT) * 0.15
        poly_coords = []
        for ang in angles:
            r = np.random.uniform(r_min, r_max)
            x = np.clip(cx + r * np.cos(ang), 0, MAP_WIDTH)
            y = np.clip(cy + r * np.sin(ang), 0, MAP_HEIGHT)
            poly_coords.append((x, y))
        
        # Generate a random weight for the AOI
        weight = float(np.random.uniform(aoi_min_weight, aoi_max_weight))
        
        # Add the AOI to the session state
        st.session_state.aois.append({"coords": poly_coords, "weight": weight})
    st.success(f"Generated {num_aois} random AOIs.")

# --- Simulation Controls ---
st.sidebar.markdown("### Simulation")
do_translate = st.sidebar.button("Initialize simulation")
simulate = st.sidebar.button("Run simulation")


# --- Manual POI Input ---
st.sidebar.markdown("### Add Custom POIs")
poi_x = st.sidebar.number_input("POI X Coordinate", min_value=0.0, max_value=MAP_WIDTH, step=1.0, value=0.0)
poi_y = st.sidebar.number_input("POI Y Coordinate", min_value=0.0, max_value=MAP_HEIGHT, step=1.0, value=0.0)
poi_weight = st.sidebar.number_input("POI Weight", min_value=0.5, max_value=20.0, step=0.5, value=1.0)

if st.sidebar.button("Add POI"):
    # Add the custom POI to the session state
    if 'poi_positions' not in st.session_state:
        st.session_state.poi_positions = []
    st.session_state.poi_positions.append((poi_x, poi_y, poi_weight))
    st.success(f"Added POI at ({poi_x}, {poi_y}) with weight {poi_weight}")

# --- Manual AOI Input ---
st.sidebar.markdown("### Add Custom AOIs")
aoi_coords = st.sidebar.text_area(
    "AOI Coordinates (e.g. `x1 y1, x2 y2, x3 y3`)",
    placeholder="10 20, 30 40, 50 60"
)
aoi_weight = st.sidebar.number_input(
    "AOI Weight", min_value=1.0, max_value=10.0, step=0.5, value=1.0, key="aoi_weight_sidebar"
)

if st.sidebar.button("Add AOI"):
    # Parse the coordinates
    try:
        coords = [
            tuple(map(float, coord.strip().split()))
            for coord in aoi_coords.split(",")
        ]
        # Validate that all entries are valid (x, y) tuples
        if any(len(coord) != 2 for coord in coords):
            raise ValueError("Each coordinate must have two values")
        if len(coords) < 3:
            st.error("AOI must have at least 3 vertices")
        else:
            # Add the AOI to the session state
            if "aois" not in st.session_state:
                st.session_state.aois = []
            st.session_state.aois.append({"coords": coords, "weight": aoi_weight})
            st.success(f"Added AOI with {len(coords)} vertices and weight {aoi_weight}")
    except Exception as e:
        st.error(f"Invalid AOI: {e}")

# --- Display AOIs ---
st.markdown("### Current AOIs")
if "aois" in st.session_state and st.session_state.aois:
    try:
        # Validate and prepare AOI data for the DataFrame
        aoi_data = [
            {
                "Vertices": len(aoi["coords"]),
                "Coordinates": [
                    coord for coord in aoi["coords"] if len(coord) == 2
                ],  # Ensure all coordinates are valid (x, y) tuples
                "Weight": aoi["weight"]
            }
            for aoi in st.session_state.aois
        ]
        aoi_df = pd.DataFrame(aoi_data)
        st.dataframe(aoi_df)
    except Exception as e:
        st.error(f"Error displaying AOIs: {e}")
else:
    st.write("No AOIs added yet.")

# --- Initialize Drone Positions ---
if ('num_drones' not in st.session_state
    or st.session_state.num_drones != num_drones
    or st.session_state.death_range != (death_min, death_max)):
    st.session_state.num_drones = num_drones
    st.session_state.death_range = (death_min, death_max)
    # random positions and death probabilities
    st.session_state.drone_positions = []
    st.session_state.drone_death_probs = []
    for _ in range(num_drones):
        x = float(np.random.uniform(0, MAP_WIDTH))
        y = float(np.random.uniform(0, MAP_HEIGHT))
        dp = float(np.random.uniform(death_min, death_max))
        st.session_state.drone_positions.append((x, y))
        st.session_state.drone_death_probs.append(dp)

# --- Initialize POI Positions ---
if 'num_pois' not in st.session_state \
   or st.session_state.num_pois != num_pois \
   or st.session_state.poi_range != (poi_min, poi_max):
    st.session_state.num_pois = num_pois
    st.session_state.poi_range = (poi_min, poi_max)
    # random POI positions and weights
    st.session_state.poi_positions = [
        (
            float(np.random.uniform(0, MAP_WIDTH)),
            float(np.random.uniform(0, MAP_HEIGHT)),
            float(np.random.uniform(poi_min, poi_max))
        )
        for _ in range(num_pois)
    ]

# --- Initialize AOIs ---
if ('num_aois' not in st.session_state
    or st.session_state.num_aois != num_aois):
    st.session_state.num_aois = num_aois
    st.session_state.aois = []


# --- Display Parameters ---
st.write(f"**Number of drones:** {num_drones}")
st.write(f"**Drone death probability range:** ({death_min}, {death_max})")
st.write(f"**Number of POIs:** {num_pois}")
st.write(f"**POI weight range:** ({poi_min}, {poi_max})")
st.write(f"**Number of Areas of Interest:** {num_aois}")

# --- Plot Drone Map ---
df = pd.DataFrame(
    st.session_state.drone_positions,
    columns=['x', 'y']
)
fig = px.scatter(
    df, x='x', y='y',
    title=f"Drone Positions on {MAP_WIDTH}×{MAP_HEIGHT} Map",
    labels={'x': 'X Coordinate', 'y': 'Y Coordinate'},
    height=600, width=800
)

# annotate drones with index and death probability
drone_texts = [
    f"Drone {i}<br>death_prob={st.session_state.drone_death_probs[i]:.3f}"
    for i in range(len(st.session_state.drone_positions))
]
fig.data[0].update(
    mode='markers+text',
    text=drone_texts,
    textposition='top center'
)

fig.update_layout(
    xaxis=dict(range=[0, MAP_WIDTH], fixedrange=True),
    yaxis=dict(range=[0, MAP_HEIGHT], fixedrange=True)
)

# # add POIs to the map
# for x, y, w in st.session_state.poi_positions:
#     poi_text = f"POI<br>weight={w:.1f}"
#     fig.add_trace(go.Scatter(
#         x=[x], y=[y],
#         mode='markers+text',
#         marker=dict(color='red', size=8),
#         text=[poi_text],
#         textposition='bottom center',
#         showlegend=False
#     ))

# st.plotly_chart(fig)

# build AOI objects
aois_objs = [
    AreaOfInterest(area['coords'], weight=area['weight'])
    for area in st.session_state.aois
]
draw_static(
    drones=[Quadcopter(pos) for pos in st.session_state.drone_positions],
    pois=[PointOfInterest(pos[:2], weight=pos[2]) for pos in st.session_state.poi_positions],
    aois=aois_objs,
    bounds=(MAP_WIDTH, MAP_HEIGHT),
    filename="online_sim/static_simulation.png", 
    streamlit_display=True
)

# Run the simulation using draw_static function
if do_translate:
    # build AOI objects
    aois_objs = [
        AreaOfInterest(area['coords'], weight=area['weight'])
        for area in st.session_state.aois
    ]
    draw_static(
        drones=[Quadcopter(pos) for pos in st.session_state.drone_positions],
        pois=[PointOfInterest(pos[:2], weight=pos[2]) for pos in st.session_state.poi_positions],
        aois=aois_objs,
        bounds=(MAP_WIDTH, MAP_HEIGHT),
        filename="online_sim/static_simulation.png", 
        streamlit_display=True
    )

# --- Run Simulation ---
if simulate:
    aois_objs = [
        AreaOfInterest(area['coords'], weight=area['weight'])
        for area in st.session_state.aois
    ]
    animate_simulation(
        drones=[Quadcopter(pos).set_death_prob(dp) for pos, dp in zip(st.session_state.drone_positions, st.session_state.drone_death_probs)],
        pois=[PointOfInterest(pos[:2], weight=pos[2]) for pos in st.session_state.poi_positions],
        aois=aois_objs,
        bounds=(MAP_WIDTH, MAP_HEIGHT),
        results_dir="online_sim",
        num_steps=100,
        seed=np.random.randint(0, 100),
        streamlit_display=True
    )
else:
    st.markdown("Click the button to run the simulation.")