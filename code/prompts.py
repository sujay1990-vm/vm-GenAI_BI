domain_system_text = """\
You are an assistant that identifies relevant domain(s) from the user query 
among the following 2 possible domains:

Here are 2 potential domains for a senior living care system, each with distinct keywords:

Census :
Captures comprehensive daily occupancy and resident tracking data by integrating resident demographics with facility, unit, location, room type, and census status details. 
Keywords : Census, Resident Count, Daily Avg Census, Number of Residents.

Medical Event :
Documents clinical incidents affecting residents such as injuries, infections, and medication errors, along with their severity and outcomes. 
Keywords : Clinical Notes, Medical Event, Medical Event type, Falls, Wounds, Skin Tear. 

Instructions:
- Analyze the user query and decide which domain(s) are relevant.
- Output a JSON object with a single key "domains" that is an array of objects.
- Each object must have two properties: "domain_name" (a string, either "Census" or "Medical Event") and "confidence" (a float between 0 and 1).
- Only include a domain in the array if its confidence is greater than 0.
- If multiple domains are applicable, list them all with appropriate confidence scores.
- If none apply, return an empty array.
"""


sql_gen_system = """\
You are a SQL expert with strong attention to detail.
Generate queries executable in Spark SQL. Use Domain Instructions to generate accurate SQL queries.
Given:
- The user query
- The recognized intent
- The following metadata for the domain (tables and their columns information):
- The Entity Relationship Between Dimension and Fact Tables
- Domain instructions and Sample SQL queries

<IMPORTANT NOTE>
Generate a syntactically correct SQL query in plain text (no triple backticks).
Return ONLY the SQL without any commentary.
</IMPORTANT NOTE>
Constraints:
- Prefer only the relevant columns rather than using SELECT *.
- Look at previous queries tried and any errors encountered, and correct them accordingly.
- If a string has multiple words (e.g., 'Part Time', 'New York'), ensure it is placed in quotes exactly as it appears, preserving spaces.
- Do NOT merge multi-word phrases into single words.
- Follow Domain Instructions strictly
"""

census_domain_instructions = """
Domain Specific Instructions:
1. Daily Average Number of Residents - Instead of averaging the ResidentKey value, calculate the daily average as:
   (Total count of residents for the period) divided by (number of days in the period).
   e.g. COUNT(ResidentKey) / 365. - Daily Average for the year
   COUNT(ResidentKey) / 31. - Daily Average for the month of March
   COUNT(ResidentKey) / 28. - Daily Average for the month of Feb
2. Calculate Age of Resident using column Current date - ResidentDateOfBirth
3. Census - when asked, Count of Residents (COUNT(ResidentKey))
4. Daily report - when asked , Generate Query group by Date. Do not Average
5. MTD - Month to Date, when asked , include all days from start of the specified month to specified date.
6. YTD - Year to Date, when asked , include all days from start of the specified Year to specified date.
"""


census_sample_sql_queries = """
Sample SQL queries : 
1. SELECT COUNT(DISTINCT ResidentKey) AS NumResidents FROM Fact_Census WHERE LocationKey = (SELECT LocationKey FROM Dim_CensusLocation WHERE LocationName = 'San Francisco')
2. SELECT AVG(ResidentKey) AS AverageResidentCount FROM Fact_Census WHERE CensusDateKey IN (    SELECT CensusDateKey    FROM Dim_CensusDate    WHERE CensusDateMonth = 3) AND LocationKey = (    SELECT LocationKey  FROM Dim_CensusLocation WHERE LocationName = 'Miami-Dade')
3. SELECT AVG(ResidentKey) AS AvgResidentCount
FROM Fact_Census
WHERE LocationKey = (SELECT LocationKey FROM .Dim_CensusLocation WHERE LocationName = 'Los Angeles')
AND CensusDateKey IN (SELECT CensusDateKey FROM Dim_CensusDate WHERE CensusDateYear = YEAR(CURRENT_DATE()))
4. SELECT AVG(CensusId) AS AvgCensus FROM Fact_Census WHERE FacilityKey IN (   (SELECT FacilityKey FROM Dim_CensusFacility WHERE FacilityName = 'Meadowbrook Place'),(SELECT FacilityKey FROM Dim_CensusFacility WHERE FacilityName = 'Willow Creek'));
"""

census_entity_relationships = """
Dim_CensusResident.ResidentKey = Fact_Census.ResidentKey (1-to-many: one resident can appear in many Fact_Census rows)
Dim_CensusFacility.FacilityKey = Fact_Census.FacilityKey (1-to-many: one facility can appear in many Fact_Census rows)
Dim_CensusUnit.UnitKey = Fact_Census.UnitKey (1-to-many: one unit can appear in many Fact_Census rows)
Dim_CensusStatus.CensusStatusKey = Fact_Census.CensusStatusKey (1-to-many: one status can appear in many Fact_Census rows)
Dim_CensusRoomType.RoomTypeKey = Fact_Census.RoomTypeKey (1-to-many: one room type can appear in many Fact_Census rows)
Dim_CensusDate.CensusDateKey = Fact_Census.CensusDateKey (1-to-many: one date record can appear in many Fact_Census rows)
Dim_CensusLocation.LocationKey = Fact_Census.LocationKey (1-to-many: one location can appear in many Fact_Census rows)
"""


census_table_metadata = """
TABLE: Dim_CensusRoomType
TYPE: Dimension
COLUMNS:
 - RoomTypeKey (int): Unique identifier assigned to a Room Type [Example: 1]
 - RoomTypeId (int): Unique identifier assigned to a Room Type [Example: 1]
 - RoomTypeCode (string): Unique code assigned to a Room Type [Example: 1]
 - RoomTypeName (string): Name of the Room Type [Example: Private Assisted Living Room, Memory Care Unit]
 - RoomTypeGroup (string): Group under which the Room comes [Example: Private, Shared, Palliative Care]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]


TABLE: Dim_CensusStatus
TYPE: Dimension
COLUMNS:
 - CensusStatusKey (int): Unique identifier assigned to a Census Status [Example: 1]
 - CensusStatusCode (string): Unique Code assigned to the Census Status [Example: A]
 - CensusStatusName (string): Description of the Census Status [Example: Active]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]

TABLE: Dim_CensusDate
TYPE: Dimension
COLUMNS:
 - CensusDateKey (int): A unique numeric identifier for the date in YYYYMMDD format. [Example: 19610414]
 - CensusDateDate (string): The actual calendar date corresponding to the record. [Example: 1961-04-14 00:00:00]
 - CensusDateDayInQuarter (int): The sequential day number within the current quarter. [Example: 14]
 - CensusDateDayName (string): The full name of the day of the week for the date. [Example: Friday]
 - CensusDateDayNameAbbrevation (string): The abbreviated name of the day of the week. [Example: Fri]
 - CensusDateDayOfMonth (int): The numeric day of the month. [Example: 14]
 - CensusDateDayOfWeek (int): The numeric representation of the day within the week. [Example: 6]
 - CensusDateDayOfWeekInMonth (int): The occurrence count of that particular weekday in the month. [Example: 3]
 - CensusDateDayOfYear (int): The day number of the year, counting from January 1. [Example: 104]
 - CensusDateFirstDayOfMonth (string): The first calendar day of the month. [Example: 1961-04-01 00:00:00]
 - CensusDateFirstDayOfQuarter (string): The starting date of the quarter. [Example: 1961-04-01 00:00:00]
 - CensusDateFirstDayofYear (string): The first day of the calendar year. [Example: 1961-01-01 00:00:00]
 - CensusDateHoliday (string): The name of the holiday if the date is a recognized holiday, or blank if not applicable. [Example: nan]
 - CensusDateIsHoliday (bool): A Boolean flag indicating whether the date is a holiday. [Example: False]
 - CensusDateIsWeekday (bool): A Boolean flag indicating if the date falls on a weekday. [Example: 1]
 - CensusDateIsWeekend (bool): A Boolean flag indicating if the date falls on a weekend. [Example: False]
 - CensusDateLastDayOfQuarter (string): The last day of the quarter in which the date falls. [Example: 1961-06-30 00:00:00]
 - CensusDateLastDayofMonth (string): The final day of the month. [Example: 1961-04-30 00:00:00]
 - CensusDateLastDayofYear (string): The last day of the calendar year. [Example: 1961-12-31 00:00:00]
 - CensusDateMonth (int): The numeric month value for the date. [Example: 4]
 - CensusDateMonthAbbrevation (string): The abbreviated name of the month. [Example: Apr]
 - CensusDateMonthName (string): The full name of the month. [Example: April]
 - CensusDateMonthOfQuarter (int): The position of the month within the current quarter. [Example: 1]
 - CensusDateQuarter (int): The numeric quarter of the year in which the date falls. [Example: 2]
 - CensusDateQuarterName (string): The textual representation of the quarter. [Example: Second]
 - CensusDateQuarterShortName (string): The abbreviated quarter name. [Example: Q2]
 - CensusDateWeekOfMonth (int): The week number within the month for the given date. [Example: 3]
 - CensusDateWeekOfQuarter (int): The week number within the quarter. [Example: 3]
 - CensusDateWeekOfYear (int): The week number of the year during which the date occurs. [Example: 15]
 - CensusDateYYYYMM (string): A concatenated representation of the year and month in YYYY/MM format. [Example: 1961/04]
 - CensusDateYear (int): The four-digit year portion of the date. [Example: 1961]
 - CensusDateYearAndQuarter (string): A combined representation of the year and quarter. [Example: 1961/Q2]
 - CensusDateYearMonth (string): A combined representation of the year and abbreviated month. [Example: 1961/Apr]
 - CensusDateYearName (string): A textual label for the calendar year. [Example: CY 1961]
 - CensusDateFirstDayOfFiscalYear (string): The first day of the fiscal year corresponding to the date. [Example: 1960-10-01 00:00:00]
 - CensusDateFiscalDateKey (int): A unique fiscal date key in a format similar to the calendar date key. [Example: 19610414]
 - CensusDateFiscalDayOfYear (int): The sequential day number within the fiscal year. [Example: 196]
 - CensusDateFiscalMonth (int): The fiscal month number for the date. [Example: 7]
 - CensusDateFiscalQuarter (int): The fiscal quarter as a numeric value. [Example: 3]
 - CensusDateFiscalQuarterName (string): The abbreviated or textual fiscal quarter name. [Example: Q3]
 - CensusDateFiscalWeekOfYear (int): The week number within the fiscal year. [Example: 28]
 - CensusDateFiscalYear (string): The fiscal year designation. [Example: FY1961]
 - CensusDateIsFirstDayOfFiscalYear (int): A flag indicating whether the date is the first day of the fiscal year. [Example: False]
 - CensusDateIsLastOfFiscalYear (int): A flag indicating whether the date is the last day of the fiscal year. [Example: False]
 - CensusDateLastDayOfFiscalYear (string): The final day of the fiscal year. [Example: 1961-09-30 00:00:00]

TABLE: Dim_CensusResident
TYPE: Dimension
COLUMNS:
 - ResidentKey (int): Unique identifier assigned to a Resident [Example: 1]
 - ResidentName (string): Name of the Resident [Example: Genevieve J Majkrzak]
 - ResidentDateOfBirth (string): Date of Birth of the Resident [Example: 12/19/1945, 1/28/1955]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]

TABLE: Dim_CensusFacility
TYPE: Dimension
COLUMNS:
 - FacilityKey (int): Unique identifier assigned to a facility [Example: 1]
 - FacilityCode (string): Unique code assigned to a Facility [Example: "08"]
 - FacilityName (string): Name of the Facility [Example: Reliant care at Los Angeles, Reliant care at Harris]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]


TABLE: Dim_CensusLocation
TYPE: Dimension
COLUMNS:
 - LocationKey (int): Unique identifier assigned to a Location [Example: nan]
 - LocationId (string): Unique identifier assigned to a Location [Example: 1]
 - LocationCode (string): Unique code assigned to a Location [Example: 4]
 - LocationName (string): Name of the Location [Example: Dallas, Los Angeles]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]

TABLE: Dim_CensusUnit
TYPE: Dimension
COLUMNS:
 - UnitKey (int): Unique identifier assigned to a Unit [Example: nan]
 - UnitId (string): Unique identifier assigned to a Unit [Example: 1]
 - UnitCode (string): Unique code assigned to a Unit [Example: 10]
 - UnitName (string): Name of the Unit [Example: Evergreen way]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]

TABLE: Fact_Census
TYPE: Fact
SOURCE: PCC
COLUMNS:
 - ResidentKey (int): Unique key from Resident dimension [Example: 3903]
 - FacilityKey (int): Unique key from Facility dimension [Example: 2]
 - UnitKey (int): Unique key from Unit dimension [Example: 1,2,3]
 - LocationKey (int): Unique key from Locaton dimension [Example: 1]
 - RoomTypeKey (int): Unique key from RoomType dimension [Example: 7,8]
 - CensusStatusKey (int): Unique key from CensusStatus dimension [Example: 1]
 - CensusDateKey (int): Unique key used for data modelling [Example: 20090102]
 - ReportDateKey (int): Unique key used for data modelling [Example: 20090101]
 - CensusId (string): Unique Identifier of a Census event [Example: 1]
 - CensusFactId (string): Unique identifier assigned to a Census record in the Fact [Example: CF1, CF2, CF3]
 - Id (string): Unique identifier for Data Modelling [Example: 1]
 - RecordStatus (int): Metadata indicating the status of the record [Example: Active]

"""


census_response_system = """\
You are a helpful assistant that translates database query results into a concise, 
natural-language response. The Answer has to be well aligned with the original user query. The answer should cover all details necessary. 
If no rows were returned, inform the user to ask the question in a different way as the SQL result was empty.
"""


medical_events_metadata = """

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
 - MedicalEventTypeKey (int): Unique identifier assigned to a MedicalEvent [Example: 1,2,etc.]
 - MedicalEventTypeName (string): Name of the Medical Event [Example: Ventilator Malfunction,Missed Dose,Bacterial Infection ,etc.]
 - MedicalEventTypeGroup (string): Group assigned to a Medical Event type [Example: Medication Error,Fall,Wound,etc.]

TABLE: Dim_MedicalEventSeverity
TYPE: Dimension
COLUMNS:
 - MedicalEventSeverityKey (int): Unique identifier assigned to a Medical Event Severity [Example: 1,2,etc.]
 - MedicalEventSeverity (string): Severity Level of the Medical Event Type [Example: Bacteremia,BLOOD CULTURE #1,etc.]

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
 - MedicalEventDateMonth (string): The numeric month of the date (e.g., 4 for April). [Example: 4]
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
 - MedicalEventDateYear (string): The calendar year [Example: 1961]
 - MedicalEventDateYearAndQuarter (string): Year and quarter combined [Example: 1961/Q2]
 - MedicalEventDateYearMonth (string): Year and month combined with month abbreviation [Example: 1961/Apr]
 - MedicalEventDateYearName (string): Calendar year prefixed [Example: CY 1961]
 - MedicalEventDateFirstDayOfFiscalYear (string): The first day of the fiscal year. [Example: 1960-10-01 00:00:00]
 - MedicalEventDateFiscalDateKey (string): The fiscal date key formatted in yyyyMMdd. [Example: 19610414]
 - MedicalEventDateFiscalDayOfYear (string): The sequential day number within the fiscal year. [Example: 196]
 - MedicalEventDateFiscalMonth (string): The month number within the fiscal year. [Example: 7]
 - MedicalEventDateFiscalQuarter (string): The quarter number within the fiscal year. [Example: 3]
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

medical_events_domain_instructions = """
Domain Specific Instructions:
1. unique MedicalEventTypes are: Bacterial Infection
Viral Infection
Fungal Infection
Urinary Tract Infection
Wrong Dosage
Missed Dose
Incorrect Medication
Administration Error
Slip and Fall
Trip Fall
Fainting Fall
Pressure Ulcer
Laceration
Abrasion
Abnormal Blood Test
High Cholesterol
Low Hemoglobin
Agitation
Wandering
Verbal Aggression
Ventilator Malfunction
IV Pump Failure
Monitoring Device Error
Food Allergy Reaction
Medication Allergy Reaction
Environmental Allergy Reaction
2. Nature of Injury:
abrasion, ulcer , wound re-opened, none
"""

medical_events_entity_relationships = """
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

medical_events_response_system = """\
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

medical_events_sample_sql_queries = """"""