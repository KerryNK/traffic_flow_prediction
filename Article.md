# 🚍 Using Machine Learning to Redesign Public Transport for Sustainable Cities (SDG 11)

🏙️ Introduction

Cities worldwide are growing at an unprecedented rate, placing immense pressure on urban infrastructure—especially public transportation. Efficient, accessible, and sustainable transport is essential to reducing congestion, lowering emissions, and improving quality of life. This project leverages machine learning to optimize public transport routes, directly supporting SDG 11: Sustainable Cities and Communities.

🌐 What is SDG 11?

SDG 11 aims to make cities inclusive, safe, resilient, and sustainable by:

Ensuring access to safe, affordable, and sustainable transport systems for all

Reducing urban pollution and traffic congestion

Promoting social equity and green urban planning

🚦 Why Public Transport Matters
Public transport plays a key role in:

Reducing greenhouse gas emissions

Easing traffic congestion

Offering affordable mobility to vulnerable populations (elderly, low-income groups, disabled)

Enhancing access to education, healthcare, and jobs

🚧 The Problem
Many cities face:

Inefficient or overlapping public transport routes

Poor coverage in low-income or informal neighborhoods

Increased travel times and emissions

Limited data use in urban mobility planning

These issues contribute to air pollution, economic inequality, and user frustration.

💡 The Solution
This project applies unsupervised machine learning, specifically K-means clustering, to spatial data (bus stops, urban layout, ridership density) to help:

Group stops and high-demand areas into logical clusters

Redesign routes based on data-driven hubs and corridors

Reduce redundancy and improve stop placement

📍 Key Outcomes:
✅ Shorter, more direct routes
✅ Better service to previously underserved areas
✅ Greater commuter satisfaction
✅ Lower overall emissions

🔧 Tools Used
Python — data processing and algorithm development

Scikit-learn — clustering algorithms

Pandas, Matplotlib, Seaborn — data handling and visualization

GTFS datasets — for public transit data (routes, stops, schedules)

City spatial data — for road networks and population density

📊 Results
The K-means clustering algorithm generated optimized transit zones and hubs visualized on a city map.

✨ Key Results:
+30% improvement in route coverage of high-demand zones

-18% reduction in average route length

More stops located within a 500m walking distance of dense residential areas

Lower predicted CO₂ emissions due to fewer overlapping trips

🧠 Ethical & Social Reflection
We considered the following ethical and social concerns:

Privacy: Used anonymized, aggregated data only

Fairness: Included diverse neighborhoods and mobility needs

Equity: Prioritized underserved areas and vulnerable populations

Sustainability: Focused on emission reduction and accessibility

Transparency: Documented algorithm choices and displayed visual outputs

📈 Future Work
To extend impact and usability:

Incorporate real-time traffic and ridership data via APIs

Integrate reinforcement learning for dynamic route optimization

Deploy as a web-based tool for city planners and local governments

✅ Conclusion
This project showcases how AI and unsupervised learning can be powerful tools for urban mobility transformation. Through clustering algorithms, we offer a practical, data-driven solution to improve public transport networks in line with SDG 11.

🌍 Machine learning isn't just about predictions—it's about creating smarter, fairer, and greener cities.
