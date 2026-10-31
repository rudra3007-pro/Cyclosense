## AI/ML-Based Tropical Cyclone Identification, Classification and Prediction Using Multi-Source Satellite Data

CycloSense — Revised Detailed Project Report

Version 2.0 — SIH 2026 Draft for Review

## Revision focus

This version preserves the original CycloSense problem statement, core AI pipeline, multi-source data strategy, GIS layer, explainability, uncertainty handling, evaluation plan, and deployment approach while incorporating three targeted project-level enhancements.

| Replace | “Ask the Storm” multilingual public | Replaced by an AI Action Planner / Decision Assistant that converts |
| --- | --- | --- |
|   | chatbot | forecast and risk outputs into location- and user-specific preparedness |
|   |   | actions. |
| Add | Cyclone Impact Forecasting | Extends the system beyond track prediction to estimate region-level |
|   |   | wind, rainfall/flood exposure, infrastructure/population exposure and |
|   |   | overall impact risk. |
| Add | What-If / Scenario Simulator | Allows users to test hypothetical track/intensity changes and compare |
|   |   | how affected regions and risk estimates change. |


## Table of Contents

1. Executive Summary

2. Problem Statement

3. Proposed Solution

3.1. Solution Overview — Core Pipeline

3.2. Enhanced Decision-Support Layer

3.3. Unique / Special Points

4. Why AI Is Required

5. High-Level System Architecture

6. Data Sources

7. Data Pipeline

8. Data Preprocessing

9. Data Synchronization and Dataset Creation

10. AI Module 1 — Cyclone Detection

11. AI Module 2 — Cyclone Pattern Classification

12. AI Module 3 — Intensity Estimation

13. AI Module 4 — Temporal Forecasting

14. Track Prediction

15. Intensity Forecast

16. Multi-Source Data Fusion

17. Explainable AI and Uncertainty

18. Cyclone Impact Forecasting — NEW

19. Geographic Risk Analysis

20. AI Action Planner / Decision Assistant — REPLACES ASK THE STORM

21. What-If / Scenario Simulator — NEW

22. Storm DNA Similarity Search

23. Frontend and GIS Visualization

24. Backend and Prediction API Flow

25. Database and Data Storage

26. Complete Software Stack

27. Evaluation Methodology

28. Minimum Viable Product (MVP)

29. Advanced Version

30. Real-Time / Operational Mode


31. Deployment Architecture

32. Hardware Requirements

33. Team Division — 6 Members

34. Development Roadmap

35. Final Demonstration Flow

36. What Makes the Revised Project Innovative

37. Important Limitations

38. Conclusion

39. Final Technology Stack Summary


## 1. Executive Summary

Tropical cyclones are among the most destructive natural hazards affecting coastal populations, infrastructure, agriculture, and maritime activity. Accurate and timely identification, classification, and prediction of cyclone behaviour are essential for disaster preparedness and mitigation.

CycloSense is an AI/ML-based Tropical Cyclone Monitoring and Prediction System that ingests multi-source satellite imagery together with meteorological and oceanic data. It detects cyclones, locates their centres, classifies structural and intensity patterns, estimates current intensity, forecasts future track and intensity, quantifies uncertainty, and communicates geographic risk through an interactive GIS dashboard.

This revised version strengthens the original design with three project-level differentiators. First, the original “Ask the Storm” chatbot is replaced by an AI Action Planner / Decision Assistant that converts model outputs into practical, role-specific preparedness guidance. Second, Cyclone Impact Forecasting extends track forecasting into region-level hazard and exposure estimates. Third, a What-If / Scenario Simulator allows users to explore hypothetical changes in cyclone track or intensity and observe their effect on predicted impact.

The system remains a decision-support and research prototype rather than an official meteorological warning service. The revised architecture is therefore designed to complement, not replace, official forecasting agencies.

## 2. Problem Statement

To develop an Artificial Intelligence (AI) / Machine Learning (ML) based system for the identification, classification, and prediction of different tropical cyclone patterns using multi-source satellite data.

Cyclone monitoring requires analysis of large volumes of satellite imagery, historical track records, and environmental parameters. Cyclone behaviour is governed by complex, non-linear spatial and temporal processes such as spiral cloud organisation, eye formation, convective bursts, and changing environmental conditions.

The revised system addresses not only detection and forecasting, but also the gap between forecast information and practical preparedness decisions. The objective is to produce transparent, location-aware outputs that can support disaster-management interpretation and public preparedness.

## 3. Proposed Solution

CycloSense is an end-to-end AI-powered Tropical Cyclone Monitoring and Prediction System that analyses satellite imagery alongside meteorological and oceanic parameters.

The core system performs detection, localisation, pattern classification, current intensity estimation, temporal analysis, future track forecasting, future intensity forecasting, uncertainty estimation, geographic risk analysis, and GIS visualisation.

The revised decision-support layer adds three enhanced capabilities: an AI Action Planner / Decision Assistant, Cyclone Impact Forecasting, and a What-If / Scenario Simulator.

Multi-Source Data Preprocessing AI Detection AI Classification Temporal Forecasting Impact &

Risk Analysis Action Planning / Scenario Analysis GIS Dashboard

## 3.1 Solution Overview — Core Pipeline

The operational pipeline is organised into six functional stages, with the decision-support layer extending the forecast outputs into impact and action-oriented products.

| Stage | Function |
| --- | --- |
| SENSE | Ingest and preprocess multi-source satellite, meteorological, and |
|   | oceanic data. |


| Stage | Function |
| --- | --- |
| DETECT | Use YOLO-based detection to identify cyclone presence and |
|   | estimate the cyclone centre. |
| CLASSIFY | Use CNN / Vision Transformer models to classify cyclone |
|   | structural or intensity patterns and estimate current intensity. |
| FORECAST | Use temporal deep learning to predict future track and intensity |
|   | and estimate forecast uncertainty. |
| MATCH | Use Storm DNA to retrieve similar historical cyclones for |
|   | case-based context. |
| VISUALIZE | Present tracks, intensity, uncertainty, impact zones, actions, and |
|   | scenarios through the GIS dashboard. |


## 3.2 Enhanced Decision-Support Layer

The decision-support layer sits after the forecasting engine. It consumes predicted track, predicted intensity, uncertainty, distance from coast, geographic exposure, and environmental information.

The layer produces three enhanced outputs: Cyclone Impact Forecasting, AI Action Planner / Decision Assistant, and What-If / Scenario Simulation.

| Enhanced Capability | Purpose | Primary Output |
| --- | --- | --- |
| Cyclone Impact Forecasting | Translate forecast hazards into | Wind / rainfall / flood risk, exposed |
|   | region-level impact and exposure | population and infrastructure, affected |
|   | estimates. | regions, impact severity. |
| AI Action Planner / Decision Assistant | Translate model outputs into practical, | Preparedness priorities, recommended |
|   | role-specific actions. | actions, shelter/evacuation guidance |
|   |   | where supporting data is available, and |
|   |   | plain-language explanations. |
| What-If / Scenario Simulator | Explore hypothetical changes to the | Alternative affected regions, exposure |
|   | forecast. | changes, risk changes, and comparative |
|   |   | scenario summaries. |

## 3.3 Unique / Special Points

The following features are retained or added to differentiate CycloSense from a conventional forecasting dashboard.

- Storm DNA Similarity Search — compares current cyclone signatures, track shape, and environmental context against historical cyclone records.

- AI Action Planner / Decision Assistant — replaces the original Ask the Storm chatbot and focuses on actionable, location- and user-specific preparedness guidance.

- Explainable AI + Uncertainty Cone — Grad-CAM explains image-based classification decisions while a confidence value and geographic uncertainty region communicate forecast uncertainty.

- Cyclone Impact Forecasting — extends predicted track and intensity into region-level hazard, exposure, and impact information.

- What-If / Scenario Simulator — enables comparison of baseline and hypothetical track/intensity scenarios.

## 4. Why AI Is Required

Traditional cyclone monitoring involves manual analysis of large amounts of satellite and meteorological information. Cyclone behaviour changes continuously and contains complex spatial and temporal patterns that are difficult to encode with fixed rules.

AI models can learn patterns directly from historical observations. The system is designed to learn spiral cloud structures, cloud organisation, eye formation, convection patterns, cloud-top temperature changes, historical movement, intensity changes, and relationships between environmental parameters and cyclone evolution.

- Pattern recognition from satellite imagery

- Temporal learning from satellite sequences

- Joint track and intensity forecasting

- Multi-source feature fusion

- Probabilistic / uncertainty-aware prediction

- Geospatial impact and exposure estimation


## 5. High-Level System Architecture

The architecture connects satellite, meteorological, oceanic, and historical track data through preprocessing and synchronisation into a multi-model AI engine. Forecast outputs feed the uncertainty model and impact/risk engines, which then support action planning, scenario simulation, the backend API, and the GIS dashboard.

Multi-Source Data Preprocessing Synchronization AI Engine (YOLO CNN/ViT Regression ConvLSTM) Track + Intensity + Pattern Uncertainty Impact Forecasting Risk / Exposure Action

Planner + Scenario Simulator FastAPI PostgreSQL/PostGIS React + Leaflet/Plotly


## 6. Data Sources

The system requires several categories of data so that the AI engine has both a visual and an environmental representation of each cyclone.

| Category | Potential Inputs |
| --- | --- |
| Satellite Data | INSAT-3D / INSAT-3DR, NOAA products, Himawari data, NASA |
|   | Earth observation datasets, and other publicly available |
|   | geostationary products where access is available. |
| Cyclone Track Data | Cyclone ID, name, timestamp, latitude, longitude, maximum |
|   | wind speed, central pressure, and intensity category. IBTrACS |
|   | may be used as a standardised historical track source. |
| Meteorological Data | Wind speed, wind direction, atmospheric pressure, temperature, |
|   | relative humidity, and vertical wind shear where available. |
| Ocean Data | Sea Surface Temperature (SST), broader ocean temperature |
|   | information, and other suitable oceanic parameters. |

## 7. Data Pipeline

Raw data from all sources passes through a common pipeline before becoming usable training data.

Satellite Sources Data Download / API Raw Data Storage Quality Checking Preprocessing

Geospatial Alignment Temporal Synchronization Feature Extraction Training Dataset

## 8. Data Preprocessing

Satellite data is processed using Python and geospatial libraries. Preprocessing is designed to reduce artefacts and maintain the geographic information required for downstream mapping.

- Image cleaning: remove corrupted, duplicate, missing, and invalid observations.

- Normalisation: scale satellite pixel values for stable neural-network training.

- Resizing: convert imagery to a fixed resolution appropriate to the selected model and available GPU resources.

- Geospatial processing: preserve latitude, longitude, projection, and geographic boundary information.

## 9. Data Synchronization and Dataset Creation

A satellite observation must be matched with the corresponding cyclone observation in time. A training record typically contains image, timestamp, cyclone ID, latitude/longitude, wind speed, pressure, intensity category, satellite source, SST, and atmospheric parameters.

The dataset is organised into images, metadata, and labels indexed by cyclone identifier. Detection, classification, and forecasting labels are kept traceable to the source cyclone and observation time.

The train/validation/test split is performed by cyclone rather than randomly by image to prevent data leakage.

## 10. AI Module 1 — Cyclone Detection

Objective: determine whether a cyclone is present in a satellite image and locate it precisely.

A YOLO-based object-detection model outputs a detection flag, bounding box, and confidence score. The bounding box is converted into a geographic cyclone-centre estimate using satellite geospatial metadata.

Satellite Image YOLO Cyclone Detected? + Bounding Box + Confidence Geographic Centre


## 11. AI Module 2 — Cyclone Pattern Classification

The region detected by YOLO is cropped and passed to a classification model. Candidate architectures include EfficientNet, ResNet, ConvNeXt, and Vision Transformer (ViT). For the first implementation, EfficientNet or ConvNeXt is a practical choice.

Actual output classes should be derived from reliable ground-truth labels, such as developmental stage or meteorological intensity categories.

Satellite Image YOLO Cyclone Crop EfficientNet / ConvNeXt / ViT Pattern / Stage Class

## 12. AI Module 3 — Intensity Estimation

Rather than predicting only a discrete category, the system estimates continuous cyclone parameters directly.

- Inputs: cyclone satellite crop, satellite-derived features, environmental parameters.

- Outputs: wind speed and central pressure.

- Evaluation: MAE, RMSE, and R².

## 13. AI Module 4 — Temporal Forecasting

A single satellite image is insufficient to understand cyclone evolution. A sequence of observations is provided to a temporal deep-learning model.

ConvLSTM is the recommended first implementation because it captures both spatial structure and temporal evolution. A Temporal Transformer is retained as an advanced alternative.

Previous Satellite Sequence Feature Fusion ConvLSTM Future Track + Future Intensity + Forecast

Uncertainty

## 14. Track Prediction

The temporal model predicts future cyclone coordinates from the current position and recent history. Predicted coordinates are passed to the GIS module for visualisation.

| Horizon | Example Predicted Latitude | Example Predicted Longitude |
| --- | --- | --- |
| Current | 13.2° N | 82.7° E |
| +6 hours | 13.7° N | 83.4° E |
| +12 hours | 14.2° N | 84.1° E |
| +24 hours | 15.3° N | 85.6° E |

## 15. Intensity Forecast

The temporal model is extended to predict future intensity alongside future position.

| Horizon | Example Predicted Wind Speed |
| --- | --- |
| Current | 100 km/h |
| +6 hours | 108 km/h |
| +12 hours | 117 km/h |
| +24 hours | 128 km/h |


## 16. Multi-Source Data Fusion

Because the problem statement explicitly calls for multi-source satellite data, the system is not limited to a single satellite feed.

Two fusion strategies are considered: image-level fusion and feature-level fusion. For the first prototype, feature-level fusion is the safer and simpler approach because satellites may differ in spatial resolution and spectral channels.

INSAT Features + NOAA Features + Himawari Features + SST + Wind/Humidity Feature Fusion

ConvLSTM Future Prediction

## 17. Explainable AI and Uncertainty

Grad-CAM is used to explain image-based classification decisions by highlighting image regions most influential to the model's prediction.

Forecast uncertainty is represented using confidence values and a geographic uncertainty region or cone that widens with forecast horizon. The system avoids presenting a single deterministic track as guaranteed.

## 18. Cyclone Impact Forecasting — NEW

Cyclone Impact Forecasting is a new project-level enhancement that extends the original geographic risk analysis. Instead of stopping at “where the cyclone may travel,” the system estimates what the forecast may mean for affected regions.

The impact engine combines predicted track and intensity with geographic exposure layers and, where data is available, environmental hazard indicators. Outputs are presented region-wise rather than as a single cyclone-wide score.

| Impact Dimension | Illustrative Output |
| --- | --- |
| Wind Hazard | Predicted wind-speed band / severity for each affected region. |
| Rainfall / Flood Exposure | Estimated rainfall or flood-risk category where suitable |
|   | environmental and geographic data are available. |
| Population Exposure | Population potentially exposed to forecast hazard zones. |
| Infrastructure Exposure | Potentially exposed roads, ports, cities, and critical infrastructure |
|   | layers. |
| Overall Impact | LOW / MODERATE / HIGH / VERY HIGH impact category with |
|   | supporting indicators. |

## 19. Geographic Risk Analysis

The predicted cyclone track is spatially overlaid on coastlines, district boundaries, states, ports, major cities, population/exposure data, and critical infrastructure. Spatial intersections produce potentially affected regions.

The risk engine consumes predicted track, predicted intensity, distance from coast, uncertainty, and geographic exposure to produce LOW, MODERATE, HIGH, or VERY HIGH risk categories.

## 20. AI Action Planner / Decision Assistant — REPLACES ASK THE STORM

The original “Ask the Storm” multilingual chatbot is replaced by an AI Action Planner / Decision Assistant. The new feature is not positioned as a generic chatbot; it is a decision-support interface grounded in structured


CycloSense outputs.

The planner converts forecast, impact, uncertainty, and geographic-risk information into practical guidance. The output can be adapted to a selected user context such as resident, fisherman, port operator, hospital, or disaster-management official, provided the required supporting data is available.

Example output structure: current risk level, expected hazard window, affected area, preparedness priorities, recommended next actions, and explanation of the evidence behind the recommendation.

| User Context | Example Action-Oriented Output |
| --- | --- |
| Resident | Preparedness priorities, affected-zone status, and available |
|   | shelter / evacuation guidance where supporting GIS data exists. |
| Fisherman / Maritime User | Coastal and marine risk status, forecast window, and return / |
|   | avoidance guidance based on system outputs. |
| Port / Infrastructure Operator | Operational risk indicators and priority preparedness actions. |
| Disaster-Management Official | Priority regions, exposed assets, scenario comparison, and |
|   | resource-planning cues. |


## 21. What-If / Scenario Simulator — NEW

The What-If / Scenario Simulator allows users to explore hypothetical changes to the forecast instead of viewing only one deterministic prediction.

A scenario may modify the projected track, intensity, or selected forecast assumption. The impact engine then recalculates affected regions, exposure, and risk so users can compare the baseline and alternative cases.

This feature is intended for decision-support and sensitivity analysis. It does not claim to create a scientifically valid new meteorological forecast merely by moving a track manually.

Baseline Forecast User Scenario Change Recalculate Impact / Exposure Compare with Baseline

Scenario Dashboard

## 22. Storm DNA Similarity Search

Storm DNA compares the current cyclone's satellite signature, track shape, and environmental context against a historical database and surfaces the closest matching past cyclones.

The historical match provides explainable, case-based context rather than replacing the numerical forecast.

## 23. Frontend and GIS Visualization

The frontend is built using React.js. The interactive GIS layer uses Leaflet, with Plotly used where suitable for intensity and temporal plots.

The dashboard displays satellite imagery, current cyclone position, historical and predicted tracks, uncertainty, risk zones, impact layers, Storm DNA matches, action-planning outputs, and scenario comparisons.

- Historical track — solid line

- Current position — marker

- Predicted track — dashed line

- Uncertainty — shaded cone / polygon

- Risk zones — colour-coded polygons

- Impact layers — region-level hazard and exposure overlays

- Scenario mode — baseline vs. hypothetical comparison

## 24. Backend and Prediction API Flow

FastAPI connects the React frontend to the AI models, impact/risk engines, and database.

React Input FastAPI Preprocessing YOLO Cyclone Crop Classifier Forecasting Model Uncertainty Impact Engine Risk / Action / Scenario Engine JSON Response React Dashboard

## 25. Database and Data Storage

PostgreSQL with PostGIS is used for structured and geospatial data. Satellite images are stored in an object/file storage layer rather than directly inside PostgreSQL.

| Table | Key Fields |
| --- | --- |
| Cyclones | cyclone_id, name, start_time, end_time, region |
| Observations | cyclone_id, timestamp, latitude, longitude, wind_speed, |
|   | pressure, satellite_source, image_path |


| Table | Key Fields |
| --- | --- |
| Predictions | cyclone_id, prediction_time, forecast_hour, predicted_latitude, |
|   | predicted_longitude, predicted_wind, confidence |
| Impacts | cyclone_id, region_id, forecast_hour, hazard_level, |
|   | exposure_level, impact_level |
| Scenarios | scenario_id, cyclone_id, baseline_reference, |
|   | modified_track/intensity parameters, recalculated_risk, |
|   | recalculated_exposure |


## 26. Complete Software Stack

Core stack to implement:

| Layer | Technologies |
| --- | --- |
| AI / ML | Python, PyTorch, YOLO, EfficientNet / ConvNeXt, ConvLSTM, |
|   | Scikit-learn |
| Satellite / GIS | xarray, Rasterio, GeoPandas, Shapely, OpenCV, Cartopy |
| Backend | FastAPI, Uvicorn |
| Frontend | React, JavaScript / TypeScript, Leaflet, Plotly |
| Database | PostgreSQL, PostGIS |
| Deployment | Docker, Docker Compose, Linux, GPU server / cloud GPU when |
|   | available |
| Development | Git, GitHub, Jupyter Notebook, VS Code, MLflow |

## 27. Evaluation Methodology

Each AI component is evaluated separately, and the overall system is compared against an appropriate baseline.

| Component | Metrics |
| --- | --- |
| Detection | Precision, Recall, mAP, IoU |
| Classification | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| Intensity Estimation | MAE, RMSE, R² |
| Track Prediction | Mean Position Error; separate 6h, 12h, and 24h errors |
| Impact Forecasting | Region-level classification agreement / error and exposure |
|   | estimation error where ground truth is available |
| Scenario Simulator | Consistency checks against the modified scenario assumptions |
|   | and comparative risk/exposure outputs; not treated as a new |
|   | meteorological forecast metric |

## 28. Minimum Viable Product (MVP)

The first working version remains a minimal but complete pipeline. The new features are layered after the core prediction path is stable.

Satellite Image YOLO Detection Cyclone Centre CNN Classification Historical Track Basic

ConvLSTM Forecast Basic GIS Dashboard

## 29. Advanced Version

After the MVP is functioning end-to-end, the following capabilities are layered on top:

- Multi-source satellite fusion

- SST and meteorological parameter integration

- Intensity forecasting

- Uncertainty cone


• Grad-CAM explainability

• Risk mapping

• Automated data ingestion

• Storm DNA similarity search

• Cyclone Impact Forecasting

• AI Action Planner / Decision Assistant

• What-If / Scenario Simulator

## 30. Real-Time / Operational Mode

Once the historical-data system works reliably, a new satellite observation can trigger automated download, preprocessing, detection, classification, forecasting, impact/risk analysis, database update, and dashboard update.

For a hackathon prototype, historical cyclone replay can be used to demonstrate the workflow convincingly rather than claiming a fully operational meteorological service without the required data access, validation, and institutional integration.

New Observation Ingestion Preprocessing AI Forecast Impact / Risk Action / Scenario

Database Update Dashboard


## 31. Deployment Architecture

The deployed system routes requests from React through FastAPI, which communicates with PostgreSQL/PostGIS and the PyTorch-based inference service. Docker packages the frontend, backend, database, and AI services into reproducible containers.

## 32. Hardware Requirements

A standard laptop is sufficient for data processing, frontend/backend development, and small-scale experiments.

A GPU is strongly recommended for training detection, classification, and forecasting models. Suitable environments include a local NVIDIA GPU, Google Colab, Kaggle, or a cloud GPU service.

## 33. Team Division — 6 Members

The revised feature set can be integrated without changing the six-member structure by expanding the responsibilities of the GIS/frontend and backend members.

| Member | Role | Responsibilities |
| --- | --- | --- |
| 1 | Data Engineer | Satellite datasets, track data, |
|   |   | preprocessing, synchronisation |
| 2 | Detection | YOLO, bounding boxes, localisation, |
|   |   | evaluation |
| 3 | Classification | CNN/ViT, pattern classification, intensity |
|   |   | classification, Grad-CAM |
| 4 | Forecasting | ConvLSTM, track prediction, intensity |
|   |   | prediction, uncertainty |
| 5 | GIS / Frontend | React, Leaflet, Plotly, impact layers, |
|   |   | Action Planner UI, scenario visualisation |
| 6 | Backend / Deployment | FastAPI, PostgreSQL/PostGIS, |
|   |   | risk/impact APIs, scenario engine, |
|   |   | Docker, integration |

## 34. Development Roadmap

The revised roadmap keeps the core AI pipeline first and introduces the three new differentiators only after the prediction and GIS foundations are stable.

| Phase | Work |
| --- | --- |
| 1 — Research | Understand cyclone datasets, accessible satellite sources, |
|   | labels, and geographical region. |
| 2 — Dataset | Download historical data, synchronize satellite and track data, |
|   | preprocess images, generate labels, create storm-level splits. |
| 3 — AI | Train YOLO, classification model, intensity regression, temporal |
|   | sequences, and ConvLSTM. |
| 4 — Fusion | Add environmental features and multiple satellite sources; |
|   | compare single-source and multi-source performance. |
| 5 — Evaluation | Calculate detection, classification, intensity, and track metrics; |
|   | compare against baseline. |


| Phase | Work |
| --- | --- |
| 6 — Application | Build FastAPI backend, React dashboard, Leaflet integration, |
|   | and database. |
| 7 — Advanced Features | Grad-CAM, uncertainty cone, risk zones, historical replay, |
|   | automated ingestion, Storm DNA, Impact Forecasting, Action |
|   | Planner, Scenario Simulator. |
| 8 — Final Demonstration | Demonstrate one complete cyclone from historical imagery |
|   | through prediction, impact analysis, action planning, and |
|   | scenario comparison. |


## 35. Final Demonstration Flow

The final demonstration should show the system as a complete decision-support workflow.

Historical Satellite Images AI Detection Cyclone Classification Current Intensity Future Track Future Intensity Uncertainty Impact Forecast Risk Map AI Action Planner What-If Scenario

Interactive Dashboard

## 36. What Makes the Revised Project Innovative

The innovation should not be framed as simply “using AI.” Its stronger and more defensible value lies in integrating multiple AI, temporal, geospatial, explainability, and decision-support components into a coherent system.

- Multi-source satellite fusion

- Spatial + temporal deep learning

- Joint track and intensity forecasting

- Explainable AI (Grad-CAM)

- Prediction uncertainty visualisation

- Geospatial risk analysis

- Cyclone Impact Forecasting — NEW

- AI Action Planner / Decision Assistant — replaces Ask the Storm

- What-If / Scenario Simulator — NEW

- Historical cyclone replay

- Storm DNA historical similarity search

- Interactive GIS decision-support dashboard

## 37. Important Limitations

The system explicitly acknowledges the following limitations:

- Satellite data may be incomplete.

- Different satellites have different resolutions and observation frequencies.

- Extreme cyclones are relatively rare, causing class imbalance.

- Cyclone tracks can change unexpectedly.

- Forecast accuracy generally decreases with longer prediction horizons.

- Real-time data availability may depend on data-provider access.

- Impact estimates depend on the quality and availability of geographic, population, infrastructure, and environmental datasets.

- Scenario simulation is a sensitivity / decision-support tool and should not be interpreted as an independently validated meteorological forecast.

- AI predictions should not be treated as official meteorological warnings.

## 38. Conclusion

CycloSense demonstrates how a multi-source satellite AI system can detect tropical cyclones, identify their structural and intensity patterns, learn their evolution from historical satellite sequences, predict future track and


intensity using temporal deep learning, estimate uncertainty and geographic risk, and present the results through an interactive GIS dashboard.

The revised system extends the original design in three targeted ways: the Ask the Storm chatbot is replaced by an AI Action Planner / Decision Assistant, Cyclone Impact Forecasting translates forecast information into region-level hazard and exposure insights, and a What-If / Scenario Simulator enables comparative decision analysis.

Together, these capabilities move CycloSense from a forecasting-only prototype toward an integrated cyclone intelligence and decision-support platform while preserving the original problem statement and keeping the system explicitly positioned as a research/decision-support prototype rather than a replacement for official meteorological services.

## 39. Final Technology Stack Summary

Python + PyTorch + YOLO + EfficientNet/ConvNeXt + ConvLSTM + xarray/Rasterio + FastAPI + PostgreSQL/PostGIS + React + Leaflet + Docker.
