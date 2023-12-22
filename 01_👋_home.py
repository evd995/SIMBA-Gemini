import asyncio
# Create a new event loop
loop = asyncio.new_event_loop()
# Set the event loop as the current event loop
asyncio.set_event_loop(loop)

import streamlit as st
from trulens_eval import Tru
import logging
import sys


# Configure LlamaIndex logging to output to stdout at DEBUG level in a single line
if 'debug_logging_configured' not in st.session_state:
    logging.basicConfig(stream=sys.stdout)
    logging.getLogger('llama_index').setLevel(logging.DEBUG)
    st.session_state.debug_logging_configured = True


st.set_page_config(
    page_title="SIMBA-Gemini Demo",
    page_icon="😸",
)

# Title of the webpage
st.title("Welcome to SIMBA: Your Digital Learning Companion")

# Using columns to place text and image side by side
col1, col2 = st.columns(2)
with col1:  # First column for the text
    st.markdown("""
    ## **Empowering Education Through Innovation**

    SIMBA (Student-Interactive Mentorship Bot Assistant) is designed to revolutionize the way teachers and students 
    interact and learn. This intelligent digital assistant is more than just a platform; it's a bridge connecting 
    educational needs with technological solutions.
    """)

with col2:  # Second column for the image
    st.image("SIMBA_img.jpeg", caption='SIMBA - Your Learning Partner')

st.markdown("---")

# Introduction and brief summary
st.markdown("""

## **How SIMBA Enhances Learning:**

- **For Teachers**: Upload and share course materials effortlessly from the teacher demo page. Create engaging 
activities and monitor student responses in real-time. Gain insights into your students' learning processes through 
intuitive analytics, allowing you to tailor your teaching for maximum effectiveness.
            
- **For Students**: Access course materials and personalized assistance anytime, anywhere, from the student demo page. 
Ask SIMBA questions about your course, receive guidance based on expert learning theories, and get help organizing 
your study plans to excel in your educational journey.
            
""")

st.markdown("---")

st.markdown("""

### **The Future of Education, Today**

At SIMBA, we believe in a future where education is more accessible, interactive, and personalized. 
Join us in embracing this future, where every student's potential is recognized and every teacher 
is empowered to inspire.

### **Discover the SIMBA Difference – Enhancing Education, One Interaction at a Time.**
""")

if "tru_student" not in st.session_state:
    tru = Tru()
    tru.reset_database()

st.sidebar.success("Select a page above.")

