kpi_lookup = {
    "measure/metric": "Description/Definition",

    "daily census": "Count of residents on a given day", 
    "average daily census": "Average of Daily census in specific time period",
    "average census by Day of Week": "Average of census by days of week in the specified time period",
    "census days": "Count of residents - for Specified Time period",


    "clinical notes": " Free-text documentation by physicians or nurses detailing resident conditions, symptoms, care plans, or interventions",
    "Fall count": "Count where Medical Event Type Name = 'Fall'",
    "Fall Rate": "Number of fall events per 1,000 resident days. Calculated as: (Fall Count / Total Census) * 1000",
    "Wound Rate": "Number of wound events per 1,000 resident days. Formula: (Wound Count / Total Census) * 1000.",
    "Infection Rate": "Number of infection events per 1,000 resident days. Formula: (Infection Count / Total Census) * 1000.",
    "More than 1 fall count": "Count of residents who had more than 1 fall within the same calendar month. e.g. Fall rate, wound rate etc. ",
    "Significant Injury": "When MedicalEventSeverity = 'Severity Level 3-Serious Injury/Damage'",
    "COVID-19 Infection Count" : "MedicalEventSeverity IN ('COVID-19', 'Covid 19')",
    "GI Infections" : "MedicalEventSeverity IN ('Clostridiodes difficile', 'Clostridium Difficle')",
    "LRT Infections" : "MedicalEventSeverity IN ('Pneumonia', 'Respiratory Infection', 'Bronchitis / Tracheobronchitis', 'Bronchitis')",
    "Non UTI Antibiotic Orders" : "OrderCategory LIKE '%Antibiotic%' AND RecordStatus != 'Inactive' (from Fact_ActiveOrderMedicineClassificationSnapshot)",
    "Non UTI Infections" : "Sum of LRT, GI, COVID-19, and Other Infections",
    "Other Infections" : "Exclude UTI, COVID-19, GI, LRT, and Lab-type events; i.e., MedicalEventSeverity NOT IN ('Urinary Tract Infection', 'URINE CULT COLONY CT', 'COVID-19', 'Covid 19', 'Clostridiodes difficile', 'Pneumonia', 'Respiratory Infection', 'Bronchitis / Tracheobronchitis', 'Bronchitis') AND MedicalEventTypeGroup NOT IN ('Lab')",
    "Average of Incident Per Days" : "AVERAGE(IncidentPerDays) from Fact_MedicalEventRate",
    "Positive Cultures" : "SUM(PositiveUTICount) from Fact_MedicalEvent",
    "UA + CS" : "SUM(UTICount) from Fact_MedicalEvent",
    "UTI Record Count" : "SUM(EventMonthlyCount) from Fact_MedicalEventRate",
    "UTIs with Antibiotic" : "COUNT where MedicalEventSeverity = 'Urinary Tract Infection' AND EventStatusReason NOT IN ('Criteria Not Met') AND Etiology != 'On Admission'",
    "Medication Errors" : "COUNT where MedicationErrorFlag = 'Y' AND LocationName != 'Monroe ADC'",
    "Psychotropic Drugs" : "MedicationFlag IN ('ANTIANXIETY', 'ANTIPSYCHOTICS', 'OTHER')"
}