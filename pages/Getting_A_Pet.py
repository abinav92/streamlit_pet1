import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
from pathlib import Path

st.set_page_config(page_title = "Getting A Pet")
st.title("Getting A Pet")
st.sidebar.success("Current Page: Getting A Pet")
st.write("Once you’ve given it some serious thought and are ready to rock the pet parent life, the next exciting step is to choose where to find your furry (or scaly or feathery) companion! Let the adventure begin! ✨")

with st.expander("Adopting a Pet"):
    st.subheader("Adopting a Pet")
    st.write("Give a homeless pet a fresh start—adopt instead of shop! Animal welfare groups have a fantastic variety of loving pets, all screened, sterilized, and ready for their forever home with you. Any adoption fees cover essential care like vaccinations and sterilization, helping each pet start healthy and happy. Open your heart, adopt, and find a best friend for life! 🐾")

    st.write("*Check out the maps below to discover where your future furry friend is waiting! Your next adventure could be just around the corner—go find your perfect pet match!*")

    df = pd.read_excel(Path.cwd() / 'Maps.xlsx')

    center_lat = 1.3522365244294143 
    center_lon = 103.82005648941153
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    # Add markers to the map
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            tooltip=row["Animal Welfare Societies"],
            popup = f"""
            <b>{row['Animal Welfare Societies']}</b><br>
            <a href="{row['Website']}" target="_blank">Visit Website</a>
            """
        ).add_to(m)

    # Display the map in Streamlit
    folium_static(m)
    
    st.write("There are many more Animal Welfare Societies without a store front. Check out the table below for complete list and their offerings.")
    # df_table = pd.read_excel(Path.cwd() / 'Table.xlsx')
    # st.write(df_table)
    data_path = Path.cwd() / 'Table.xlsx'
    df = pd.read_excel(data_path)
    df['url'] = df.apply(lambda row: f'<a target="_blank" href="{row["Links"]}">{row["Animal Welfare Societies"]}</a>', axis=1)
    df_display = df[['url', 'Animals']].rename(columns={'url': 'Animal Welfare Societies', 'Animals': 'Available Animals'})
    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)


with st.expander("Buying a Pet"):
    st.subheader("Buying a Pet")
    st.write("Before you dive into pet ownership, let’s make sure you’re getting your new buddy from a top-notch pet shop! 🐶✨ Check out their ratings and reviews to ensure you’re choosing a place that prioritizes pet wellness. Remember, responsible shops follow all the necessary rules (and steer clear of those unlicensed dog farms!).")
    st.markdown(
        """
        Look for these signs of a healthy pet:
        - Bright eyes, perky ears, and a clean nose!
        - Natural, comfy movement—no awkward waddles here!
        - A shiny coat with no bald patches.
        - No excessive scratching or chewing.
        - Free from pesky fleas and ticks!
        """)
    
    st.markdown("""
    Plus, be ready for a little screening process. You’ll fill out the Pet Purchase Declaration [PPD](https://www.nparks.gov.sg/-/media/avs/resources/pet-shop/pet-purchase-declaration-form_avs.pdf) form with the retailer, and if you’re adopting a dog, you’ll also complete the online PPD via the Pet Animal Licensing System [PALS](https://pals.avs.gov.sg/) when the retailer transfers ownership.<br>
    """, unsafe_allow_html=True)
    
    st.write("Get ready to welcome your new furry friend the right way! 🐾❤️")
   