quality_table_metadata = """

TABLE: Dim_Resident
TYPE: Dimension
COLUMNS:
 - ResidentKey (int): Unique identifier assigned to a Resident [Example: 1,2 etc.]
 - ResidentName (string): Name of the Resident [Example: Raymond Mims,A Guazzelli etc.]
 - ResidentDateOfBirth (string): Date of Birth of the Resident [Example: 12/19/1945, 1/28/1955 ,etc.] 
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 09:36:15,2025-02-11 09:36:15,etc.]

TABLE: Dim_Facility
TYPE: Dimension
COLUMNS:
 - FacilityKey (int): Unique identifier assigned to a facility [Example: 1]
 - FacilityCode (string): Unique code assigned to a Facility [Example: "08"]
 - FacilityName (string): Name of the Facility [Example: Reliant care at Los Angeles, Reliant care at Harris]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]

TABLE: Dim_EHRLocation
TYPE: Dimension
COLUMNS:
 - LocationKey (int): Unique identifier assigned to a Location [Example: 1,2,etc.]
 - LocationId (string): Unique identifier assigned to a Location [Example: 1,3,etc.]
 - LocationCode (string): Unique code assigned to a Location [Example: 1,2,etc.]
 - LocationName (string): Name of the Location [Example: Dallas, Los Angeles]

TABLE: Dim_Unit
TYPE: Dimension
COLUMNS:
 - UnitKey (int): Unique identifier assigned to a Unit [Example: 1,2,etc.]
 - UnitId (string): Unique identifier assigned to a Unit [Example: 1,2,etc.]
 - UnitCode (string): Unique code assigned to a Unit [Example: 1,2,etc.]
 - UnitName (string): Name of the Unit [Example: Evergreen way, Rosewood Gardens]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]

TABLE: Dim_MedicalEventType
TYPE: Dimension
COLUMNS:
 - MedicalEventTypeKey (int): **MedicalEventTypeKey**: A unique integer identifier for each type of medical event (e.g., check-up, fall, medication error) in the assisted living facility. Used to join with fact tables, group events by type, or filter reports on specific medical event categories. [Example: ['1', '6', '3', '5', '9']]
 - MedicalEventTypeName (string): **MedicalEventTypeName**: Describes the category of a medical event (e.g., "Good Catch," "Wound," "Viral") related to residents or staff. Used to classify incidents for reporting, filtering, and grouping in analyses of health trends or care outcomes across facilities. [Example: ['Good Catch', 'Wound', 'Viral', 'Unknown', 'Bacterial']]
 - MedicalEventTypeGroup (string): **MedicalEventTypeGroup**: Categorizes medical events (e.g., "Fall," "Infection") to group similar incidents for residents across facilities. Useful for filtering, grouping, and analyzing trends in resident health and safety events. [Example: ['Good Catch', 'Wound', 'Infection', 'Medication error', 'Fall']]

TABLE: Dim_MedicalEventSeverity
TYPE: Dimension
COLUMNS:
 - MedicalEventSeverityKey (int): **MedicalEventSeverityKey**: A unique integer identifier for the severity level of a medical event. Used to join with fact tables or filter/group events by severity in the context of resident health incidents. [Example: ['31', '34', '28', '26', '27']]
 - MedicalEventSeverity (string): **MedicalEventSeverity**: This column contains descriptive labels indicating the severity or type of medical events affecting residents (e.g., illnesses, injuries). Values range from specific conditions (e.g., Covid 19) to severity levels (e.g., Severity Level 3-Serious Injury/Damage). Useful for filtering, grouping, or analyzing resident health trends and incident impact across facilities. [Example: ['Impetigo', 'Severity Level 3-Serious Injury/Damage', 'Osteomyelitis', 'Covid 19', 'Infective otitis externa']]

TABLE: Dim_EventStatus
TYPE: Dimension
COLUMNS:
 - EventStatusKey (int): Unique identifier assigned to an Event status [Example: 1,2,etc.]
 - EventStatusName (string): Name of the Event Status [Example: Closed,Suspected,etc.,]
 - EventStatusReason (string): Reason for closing the Medical Event [Example: Criteria Not Met,Discharged,etc.]

TABLE: Dim_MedicalEventDate
TYPE: Dimension
SOURCE: PCC
COLUMNS:
 - MedicalEventDateKey (int): A surrogate key representing the date in yyyyMMdd format. [Example: 19610414]
 - MedicalEventDateDate (string): The actual date in mm-dd-yyyy format. [Example: 1961-04-14 00:00:00]
 - MedicalEventDateDayInQuarter (string): The sequential day number within the quarter. [Example: 14]
 - MedicalEventDateDayName (string): The full name of the day of the week [Example: Friday]
 - MedicalEventDateDayNameAbbrevation (string): The abbreviated form of the day name [Example: Fri]
 - MedicalEventDateDayOfMonth (string): The day number within the month. [Example: 14]
 - MedicalEventDateDayOfWeek (string): Numeric representation of the day of the week (1=Monday, 7=Sunday). [Example: 6]
 - MedicalEventDateDayOfWeekInMonth (string): The occurrence of the day within the month [Example: 3]
 - MedicalEventDateDayOfYear (string): The day number within the year. [Example: 104]
 - MedicalEventDateFirstDayOfMonth (string): The first day of the month in dd-MM-yyyy format. [Example: 1961-04-01 00:00:00]
 - MedicalEventDateFirstDayOfQuarter (string): The first day of the respective quarter in mm-dd-yyyy format. [Example: 1961-04-01 00:00:00]
 - MedicalEventDateFirstDayofYear (string): The first day of the calendar year in mm-dd-yyyy format. [Example: 1961-01-01 00:00:00]
 - MedicalEventDateHoliday (string): Name of the holiday if applicable; blank if not a holiday. [Example: nan]
 - MedicalEventDateIsHoliday (string): Boolean indicating whether the date is a holiday (TRUE/FALSE). [Example: False]
 - MedicalEventDateIsWeekday (string): Boolean indicating whether the date is a weekday (TRUE/FALSE). [Example: True]
 - MedicalEventDateIsWeekend (string): Boolean indicating whether the date is a weekend (TRUE/FALSE). [Example: False]
 - MedicalEventDateLastDayOfQuarter (string): The last day of the respective quarter. [Example: 1961-06-30 00:00:00]
 - MedicalEventDateLastDayofMonth (string): The last day of the respective month. [Example: 1961-04-30 00:00:00]
 - MedicalEventDateLastDayofYear (string): The last day of the calendar year. [Example: 1961-12-31 00:00:00]
 - MedicalEventDateMonth (int): The numeric month of the date (e.g., 4 for April). [Example: 4]
 - MedicalEventDateMonthAbbrevation (string): Abbreviation of the month [Example: Apr]
 - MedicalEventDateMonthName (string): Full name of the month [Example: April]
 - MedicalEventDateMonthOfQuarter (string): The month’s position within the quarter. [Example: True]
 - MedicalEventDateQuarter (string): The numeric quarter [Example: 2]
 - MedicalEventDateQuarterName (string): Full name of the quarter [Example: Second]
 - MedicalEventDateQuarterShortName (string): Abbreviated quarter name [Example: Q2]
 - MedicalEventDateWeekOfMonth (string): Week number within the month. [Example: 3]
 - MedicalEventDateWeekOfQuarter (string): Week number within the quarter. [Example: 3]
 - MedicalEventDateWeekOfYear (string): ISO week number within the year. [Example: 15]
 - MedicalEventDateYYYYMM (string): Year and month concatenated in yyyy/MM format. [Example: 1961/04]
 - MedicalEventDateYear (int): The calendar year [Example: 1961]
 - MedicalEventDateYearAndQuarter (string): Year and quarter combined [Example: 1961/Q2]
 - MedicalEventDateYearMonth (string): Year and month combined with month abbreviation [Example: 1961/Apr]
 - MedicalEventDateYearName (string): Calendar year prefixed [Example: CY 1961]
 - MedicalEventDateFirstDayOfFiscalYear (string): The first day of the fiscal year. [Example: 1960-10-01 00:00:00]
 - MedicalEventDateFiscalDateKey (string): The fiscal date key formatted in yyyyMMdd. [Example: 19610414]
 - MedicalEventDateFiscalDayOfYear (string): The sequential day number within the fiscal year. [Example: 196]
 - MedicalEventDateFiscalMonth (int): The month number within the fiscal year. [Example: 7]
 - MedicalEventDateFiscalQuarter (int): The quarter number within the fiscal year. [Example: 3]
 - MedicalEventDateFiscalQuarterName (string): The fiscal quarter name [Example: Q3]
 - MedicalEventDateFiscalWeekOfYear (string): The week number within the fiscal year. [Example: 28]
 - MedicalEventDateFiscalYear (string): The fiscal year name prefixed with FY  [Example: FY1961]
 - MedicalEventDateIsFirstDayOfFiscalYear (string): Boolean indicator (0/1) for whether the date is the first day of the fiscal year. [Example: False]
 - MedicalEventDateIsLastOfFiscalYear (string): Boolean indicator (0/1) for whether the date is the last day of the fiscal year. [Example: False]
 - MedicalEventDateLastDayOfFiscalYear (string): The last day of the fiscal year. [Example: 1961-09-30 00:00:00]

TABLE: Dim_ReportDate
TYPE: Dimension
COLUMNS:
 - ReportDateKey (int): A surrogate key representing the date in yyyyMMdd format. [Example: 19610414]
 - ReportDateDate (string): The actual date in mm-dd-yyyy format. [Example: 1961-04-14 00:00:00]
 - ReportDateDayInQuarter (string): The sequential day number within the quarter. [Example: 14]
 - ReportDateDayName (string): The full name of the day of the week [Example: Friday]
 - ReportDateDayNameAbbrevation (string): The abbreviated form of the day name [Example: Fri]
 - ReportDateDayOfMonth (string): The day number within the month. [Example: 14]
 - ReportDateDayOfWeek (string): Numeric representation of the day of the week (1=Monday, 7=Sunday). [Example: 6]
 - ReportDateDayOfWeekInMonth (string): The occurrence of the day within the month [Example: 3]
 - ReportDateDayOfYear (string): The day number within the year. [Example: 104]
 - ReportDateFirstDayOfMonth (string): The first day of the month in dd-MM-yyyy format. [Example: 1961-04-01 00:00:00]
 - ReportDateFirstDayOfQuarter (string): The first day of the respective quarter in mm-dd-yyyy format. [Example: 1961-04-01 00:00:00]
 - ReportDateFirstDayofYear (string): The first day of the calendar year in mm-dd-yyyy format. [Example: 1961-01-01 00:00:00]
 - ReportDateHoliday (string): Name of the holiday if applicable; blank if not a holiday. [Example: nan]
 - ReportDateIsHoliday (string): Boolean indicating whether the date is a holiday (TRUE/FALSE). [Example: False]
 - ReportDateIsWeekday (string): Boolean indicating whether the date is a weekday (TRUE/FALSE). [Example: True]
 - ReportDateIsWeekend (string): Boolean indicating whether the date is a weekend (TRUE/FALSE). [Example: False]
 - ReportDateLastDayOfQuarter (string): The last day of the respective quarter. [Example: 1961-06-30 00:00:00]
 - ReportDateLastDayofMonth (string): The last day of the respective month. [Example: 1961-04-30 00:00:00]
 - ReportDateLastDayofYear (string): The last day of the calendar year. [Example: 1961-12-31 00:00:00]
 - ReportDateMonth (string): The numeric month of the date (e.g., 4 for April). [Example: 4]
 - ReportDateMonthAbbrevation (string): Abbreviation of the month [Example: Apr]
 - ReportDateMonthName (string): Full name of the month [Example: April]
 - ReportDateMonthOfQuarter (string): The month’s position within the quarter. [Example: True]
 - ReportDateQuarter (string): The numeric quarter [Example: 2]
 - ReportDateQuarterName (string): Full name of the quarter [Example: Second]
 - ReportDateQuarterShortName (string): Abbreviated quarter name [Example: Q2]
 - ReportDateWeekOfMonth (string): Week number within the month. [Example: 3]
 - ReportDateWeekOfQuarter (string): Week number within the quarter. [Example: 3]
 - ReportDateWeekOfYear (string): ISO week number within the year. [Example: 15]
 - ReportDateYYYYMM (string): Year and month concatenated in yyyy/MM format. [Example: 1961/04]
 - ReportDateYear (string): The calendar year [Example: 1961]
 - ReportDateYearAndQuarter (string): Year and quarter combined [Example: 1961/Q2]
 - ReportDateYearMonth (string): Year and month combined with month abbreviation [Example: 1961/Apr]
 - ReportDateYearName (string): Calendar year prefixed [Example: CY 1961]
 - ReportDateFirstDayOfFiscalYear (string): The first day of the fiscal year. [Example: 1960-10-01 00:00:00]
 - ReportDateFiscalDateKey (string): The fiscal date key formatted in yyyyMMdd. [Example: 19610414]
 - ReportDateFiscalDayOfYear (string): The sequential day number within the fiscal year. [Example: 196]
 - ReportDateFiscalMonth (string): The month number within the fiscal year. [Example: 7]
 - ReportDateFiscalQuarter (string): The quarter number within the fiscal year. [Example: 3]
 - ReportDateFiscalQuarterName (string): The fiscal quarter name [Example: Q3]
 - ReportDateFiscalWeekOfYear (string): The week number within the fiscal year. [Example: 28]
 - ReportDateFiscalYear (string): The fiscal year name prefixed with FY  [Example: FY1961]
 - ReportDateIsFirstDayOfFiscalYear (string): Boolean indicator (0/1) for whether the date is the first day of the fiscal year. [Example: False]
 - ReportDateIsLastOfFiscalYear (string): Boolean indicator (0/1) for whether the date is the last day of the fiscal year. [Example: False]
 - ReportDateLastDayOfFiscalYear (string): The last day of the fiscal year. [Example: 1961-09-30 00:00:00]

TABLE: Fact_MedicalEvent
TYPE: Fact
COLUMNS:
 - ResidentKey (int): Unique key from Resident dimension [Example: 1, 2 , etc.]
 - FacilityKey (int): Unique key from Facility dimension [Example: 3, 5 , etc.]
 - UnitKey (int): Unique key from Unit dimension [Example: 62, 86 , etc.]
 - LocationKey (int): Unique key from Location dimension [Example: 3, 5 , etc.]
 - MedicalEventTypeKey (int): Unique key from Medical Event Type dimension [Example: 3, 6, etc.]
 - MedicalEventSeveritykey (int): Unique key from Medical Event Severity dimension [Example: 6, 47 , etc.]
 - EventStatusKey (int): Unique key from Event Status dimension [Example: 1,2 , etc.]
 - ReportDateKey (int): Unique key from Report Date dimension [Example: 20210801, 20230401 , etc.]
 - MedicalEventDateKey (int): Unique key from Medical Event Date dimension [Example: 20210810, 20230421 , etc.]
 - EventId (string): Unique identifier assigned to a Event [Example: 10, 100 , etc.]
 - MedicalEventFactId (string): Unique identifier assigned to a Medical Event in the Fact [Example: 1, 2 , etc.]
 - MedicalEventDetail (string): Specific type of the Medical Event [Example: COLONY COUNT: 100,000+ GRAM-POSITIVE COCCI IN CHAINS, COLONY COUNT: 50,000 MIXED FLORA - THREE OR MORE SPECIES PRESENT, ISOLATION OF THREE OR MORE DIFFERENT BACTERIA IS PLEASE REPEAT IF CLINICALLY INDICATED , etc.]
 - Evaluation (string): Outcome or result of the medical event evaluation [Example: Positive, Negative , etc.]
 - ClinicalNotes (string): Detailed clinical notes associated with the medical event [Example: UTI Antibiotic, wound measuring 2cm X 1cm, etc.]
 - Etiology (string): Cause or origin of the medical event [Example: On Admission,In House,etc.]
 - Prescription (string): Medication or treatment order prescribed to a patient [Example: Molnupiravir Oral Capsule 200 MG (Aug 22, 2024 - Aug 27, 2024),Flomax Capsule 0.4 MG (Dec 07, 2024 - Indefinite),etc.]
 - MedicalEventRoom (string): Physical area within the facility where IPC measures are assessed [Example: cafeteria,activity room,E110,etc.]
 - MedicalEventTime (string): Timestamp indicating when the medical event occurred [Example:  9:50:00 AM,  4:50:00 PM , etc.]
 - InjuryFlag (string): Indicator specifying whether an injury has occurred [Example: Y,N]
 - WoundFlag (string): Indicator specifying whether a wound has occurred [Example: Y,N]
 - MedicationErrorFlag (string): Indicator specifying whether a Medication Error has occurred [Example: Y,N]
 - NatureOfInjury (string): Describes the details of Injury [Example: wound re-opened,ulcer,etc.]
 - DegreeOfInjury (string): Describes severity of the Injury [Example: Moderate Injury, Treatment Required,Mild Injury, First Aid Required,etc.]
 - SafetyPrecautionTaken (string): Safety precautions taken to minimize or avoid the risks of Injury [Example: wheelchair brakes on,walker available|others,etc.]
 - Organism (string): Details of the micro organism detected in the laboratory test [Example: ENTEROCOCCUS FAECIUM , PROTEUS MIRABILIS ENTEROCOCCUS FAECALIS , etc.]

 """

quality_domain_instructions = """
Domain Specific Instructions:
1. unique MedicalEventTypes are: Bacterial Infection, Ventilator Malfunction, Missed Dose, Medication Error, Fall, Wound, etc.
2. Nature of Injury:
abrasion, ulcer , wound re-opened, none
"""

quality_entity_relationships = """
Dim_Resident.ResidentKey = Fact_MedicalEvent.ResidentKey (1-to-many: one resident can appear in many Fact_MedicalEvent rows)
Dim_Facility.FacilityKey = Fact_MedicalEvent.FacilityKey (1-to-many: one facility can appear in many Fact_MedicalEvent rows)
Dim_EHRLocation.LocationKey = Fact_MedicalEvent.LocationKey (1-to-many: one location can appear in many Fact_MedicalEvent rows)
Dim_Unit.UnitKey = Fact_MedicalEvent.UnitKey (1-to-many: one unit can appear in many Fact_MedicalEvent rows)
Dim_MedicalEventType.MedicalEventTypeKey = Fact_MedicalEvent.MedicalEventTypeKey (1-to-many: one event type can appear in many Fact_MedicalEvent rows)
Dim_MedicalEventSeverity.MedicalEventSeverityKey = Fact_MedicalEvent.MedicalEventSeverityKey (1-to-many: one severity can appear in many Fact_MedicalEvent rows)
Dim_EventStatus.EventStatusKey = Fact_MedicalEvent.EventStatusKey (1-to-many: one event status can appear in many Fact_MedicalEvent rows)
Dim_ReportDate.ReportDateKey = Fact_MedicalEvent.ReportDateKey (1-to-many: one report date can appear in many Fact_MedicalEvent rows)
Dim_MedicalEventDate.MedicalEventDateKey = Fact_MedicalEvent.MedicalEventDateKey (1-to-many: one medical event date can appear in many Fact_MedicalEvent rows)
"""

quality_response_system = """\
You are a helpful assistant that translates database query results into a detailed, 
natural-language response. The Answer has to be well aligned with the original user query. The answer should cover all details necessary. 
If no rows were returned, inform the user to ask the question in a different way as the SQL result was empty.
<Notes>:
Summarize Clinical Notes with maximum details.
- Injuries: 
- Intervention:
- Doctor informed:
- Event summary:
"""

quality_sample_sql_queries = """
1. Show a chart for Different types of falls in 2024 ? 
Answer :
SELECT 
    met.MedicalEventTypeGroup AS FallType,
    COUNT(fme.MedicalEventFactId) AS FallCount
FROM 
    Fact_MedicalEvent fme
JOIN 
    Dim_MedicalEventType met ON fme.MedicalEventTypeKey = met.MedicalEventTypeKey
JOIN 
    Dim_MedicalEventDate med ON fme.MedicalEventDateKey = med.MedicalEventDateKey
WHERE 
    met.MedicalEventTypeGroup = 'Fall'
    AND med.MedicalEventDateYear = 2024
GROUP BY 
    met.MedicalEventTypeGroup;
"""