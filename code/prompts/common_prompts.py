targetted_domain_prompt = """
1. Quality
Definition:
The Quality domain captures resident-level medical activity and physician-led interventions that reflect the acuity and clinical risk of residents. It includes structured data and free-text documentation such as clinical notes, hospital transfers, physician orders, and medical events like falls or infections. This domain focuses on medical oversight, incident tracking, and treatment-related activity.
It includes:
Clinical Notes: Free-text documentation by physicians or nurses detailing resident conditions, symptoms, care plans, or interventions.
Medical Events: Recorded clinical incidents like falls, wounds, infections, or medication errors that indicate a need for medical attention.
Physician Orders: Orders issued by physicians related to treatments, medications, or care directives for residents.
Hospital Transfers: Resident transfers to acute care or emergency settings initiated or authorized by physicians, reflecting elevated medical acuity.
Use Cases:
Used to monitor resident health status, identify medical risks, track incidents and interventions, and summarize daily clinical activity. Also supports quality audits and medical decision-making.
Used with Census data to calculate incident rates (e.g., fall rate per 1,000 resident days)
Keywords:
Physician Orders, Medical Events, Falls, Wounds, Infections, Injuries, Acute Care, Emergency Visits, Order Category, Care Risk, Clinical Incidents, Clinical Notes, prescriptions, Fall Rate, Wound Rate, Infection Rate, Medical Event Rate

2. Census (Occupancy & Capacity Planning)
Definition:
The Census domain tracks the daily presence, movement, and room assignments of residents across units and facilities. It also includes long-term bed allocation, unit capacity planning, and forecasting tools to help manage facility utilization, resident admissions, and resource planning.
It includes:
Census: Daily resident counts by location, status (admitted/discharged), and room/bed configuration
Annual Capacity and Budget: Facility-level capacity planning, resource allocation, and occupancy forecasting for budgeting purposes
Use Cases:
Census data is used for calculating average daily census (ADC), tracking bed utilization, forecasting staffing needs, and aligning resident flow with room availability. It's also critical for compliance, operational efficiency, and strategic growth planning.
Used with Quality data to normalize medical incident rates (e.g., falls per 1,000 resident days)
Keywords:
Resident Count, Census Status, Room Type, Facility Capacity, Unit Capacity, Bed Allocation, Occupancy Forecast, Budget Date, Healthcare Planning, Census Date """

reformulation_decision_prompt = """
You are an assistant that decides whether a new user question depends on prior conversation.

Answer **True** ONLY if:
- The new query is vague or ambiguous (e.g., "what about the next day?", "and him?", "how many in that case?")
- The new query clearly follows up on a previous one (e.g., asks for more detail, next step, related metric, clarification, etc.)
- It uses vague pronouns or references (this, that, those, he, she, it)

Answer **False** if:
- The new query is **self-contained and understandable on its own**
- It asks about a **new topic, different metric, or different entity** than recent history
- It contains **clear entities or time references** (e.g., names, dates, locations)
"""

metric_resolver_prompt = """
You are a senior care KPI resolver for assisted living and long-term care environments.

Your task is to identify business metrics or KPIs mentioned in a user's query and describe them precisely.

For each identified metric:
- Provide a **plain English formula** that explains how the metric is calculated (DO NOT write SQL).
- Assign a `domain_name`, which can be either:
  - A single domain or multiple domains
- Use the provided domain descriptions to help you decide.

Strict Rules:
For each identified metric:
- Use the KPI dictionary when possible to define the term.
- Be logically accurate and cautious — avoid fabricating unknown terms

Also:
- Set visualize = true only if the user explicitly requests a chart or graph using terms like: 'chart', 'visualize', 'graph', 'plot', 'bar chart', 'line graph'. 
- Do not infer visualization intent based only on phrases like 'by month', 'per day', or 'over time'.
- If they ask for a chart or visual explicitly (e.g., 'graph', 'bar chart', 'visualize'), set visualize = true.
- Extract any specific visual instructions if present (e.g., 'by facility', 'over time').
- Extract structured filters from the user's question if mentioned. These include:
  - location: name of facility or location (e.g., 'Somerset')
  - year: year mentioned (e.g., 2023, 2022)
  - month: month mentioned (e.g., March, July)
  - quarter: quarter mentioned or implied (e.g., 'Q1', 'first quarter')
  - day_of_week: set to true if the user requests a breakdown by day of week (e.g., 'per weekday', 'compare Mondays')
  - time_granularity: set this to values like 'daily', 'weekly', 'monthly', or 'by_day_of_week' if such a grouping is implied
  - breakdown_by: if the user asks for comparison or grouping (e.g., 'by gender', 'for each unit'), include those dimensions here as a list (e.g., ['gender', 'unit'])

- Only include a filter if it is clearly stated or strongly implied. If any filters are missing, omit them from the JSON.

Return structured JSON for your response.
"""

reformulation_prompt = """
Given the chat history and the latest user question, which might reference context in the chat history, 
reformulate the question into a standalone question that can be understood without the chat history. 
If the question is related to the most recent queries or requires context from the chat history to be understood, 
include that context in the reformulated question. Do NOT answer the question; just provide the reformulated version.
"""