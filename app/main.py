import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
  data = pd.read_csv("data/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


def add_sidebar():
  st.sidebar.header("Histopathological Variability")
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean()),
      
    )
    
  return input_dict


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  
  
def get_radar_chart_mean(input_data):
    
    input_data = get_scaled_values(input_data)
  
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value',
        line_color='blue'
  ))
    
    fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
    return fig
  
  
def get_radar_chart_standard(input_data):
    
    
    input_data = get_scaled_values(input_data)
  
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error',
        line_color='skyblue'
  ))
    fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
    return fig
    
    
    
def get_radar_chart_worst(input_data):

    input_data = get_scaled_values(input_data)
  
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()
  
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value',
        line_color='red'
  ))
    
    
    fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
    return fig
    


def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value',
        line_color='blue'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error',
        line_color='skyblue'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value',
        line_color='red'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  scaler = pickle.load(open("model/scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
  
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")



# def main():
#   st.set_page_config(
#     page_title="Mammary neoplasm",
#     page_icon=":female-doctor:",
#     layout="wide",
#     initial_sidebar_state="expanded"
#   )
  
#   with open("assets/style.css") as f:
#     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
#   input_data = add_sidebar()
  
#   with st.container():
#     st.title("Mammary neoplasmr")
#     st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
  
#   col1, col2,col3,col4,col5 = st.columns([4,1])
  
#   with col1:
#     radar_chart_mean = get_radar_chart_mean(input_data)
#     st.plotly_chart(radar_chart_mean)
#     with col2:
#         radar_chart_standard = get_radar_chart_standard(input_data)
#         st.plotly_chart(radar_chart_standard)
#     with col3:
#         radar_chart_worst = get_radar_chart_worst(input_data)
#         st.plotly_chart(radar_chart_worst)
#     with col4:
#         radar_chart = get_radar_chart(input_data)
#         st.plotly_chart(radar_chart)
#     with col5:
#         add_predictions(input_data)


 
# if __name__ == '__main__':
#   main()
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    st.title("Mammary neoplasm")
    st.write("Breast cancer symptoms can manifest in various ways, and awareness of these signs is crucial for early detection and intervention.")
    st.write("1. Lump or Mass: The presence of a palpable lump or mass in the breast, distinguishable from the surrounding tissue.")
    st.write("2. Changes in Size or Shape: Observable alterations in the size or shape of the breast, including asymmetry between breasts.") 
    st.write("3. Skin Changes: Changes in the skin texture, such as redness, dimpling, or puckering, resembling the peel of an orange.")
    st.write("4. Nipple Changes: Changes in the nipple, such as inversion, retraction, or changes in direction.")
    st.write("5. Nipple Discharge: Unexplained nipple discharge, especially if it is bloody. ")
    st.write("6. Pain or Discomfort: Some individuals may experience discomfort or pain in the breast or nipple area.")
    st.write("7. Swelling: Swelling or enlargement of one breast without an apparent cause.")
    st.write("8. Lymph Node Changes: Swelling or changes in the lymph nodes under the arm or around the collarbone, indicating potential spread.")
    col1, col2 = st.columns([4, 1])

    #with col1:
    radar_chart_mean = get_radar_chart_mean(input_data)
    st.plotly_chart(radar_chart_mean)
    
    radar_chart_standard = get_radar_chart_standard(input_data)
    st.plotly_chart(radar_chart_standard)
    
    radar_chart_worst = get_radar_chart_worst(input_data)
    st.plotly_chart(radar_chart_worst)
    
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
    add_predictions(input_data)

    #with col2:

    
       


if __name__ == '__main__':
    main()
