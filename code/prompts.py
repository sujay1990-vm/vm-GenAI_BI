domain_system_text = """\
You are an assistant that identifies relevant domain(s) from the user query 
among the following 3 possible domains:

Here are 3 potential domains for a senior living care system, each with distinct keywords:

Physician Order : 
Captures detailed information on medical orders issued for residents, including order classifications, schedules, and timestamps. 
Integrates resident, physician, facility, and location identifiers to enable analysis of ordering trends and compliance.

Census :
Captures comprehensive daily occupancy and resident tracking data by integrating resident demographics with facility, unit, location, room type, and census status details. 
It leverages extensive date and time metadata (including census and report dates) to enable detailed analysis of occupancy trends, resource utilization, and operational performance.

Medical Event :
Documents clinical incidents affecting residents such as injuries, infections, and medication errors, along with their severity and outcomes. 
Combines comprehensive event details with resident, facility, unit, and location information to support robust clinical and quality-of-care analysis.

Note: 
1. If multiple domains apply, list them all in the 'domains' array.
2. Give atleast one domain name that is highly likely, don't keep the Domain name array empty 

If a domain is not relevant, do not include it or set its confidence to 0. 
If multiple domains apply, include them all with appropriate confidence 
(0 < confidence <= 1).
If none apply, return an empty array.
"""


sql_gen_system = """\
You are a SQL expert with strong attention to detail.
Generate queries executable in Spark SQL.
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
"""

census_domain_instructions = """
Domain Specific Instructions:
1. “Calculate resident count using COUNT(ResidentKey) for the given time period, then compute the average of these counts if asked by dividing by the time period. 
Do not use AVG(ResidentKey), which would simply average the numeric primary key values.”
2. Daily Average Number of Residents - Instead of averaging the ResidentKey value, calculate the daily average as:
   (Total count of residents for the period) divided by (number of days in the period).
   e.g. COUNT(ResidentKey) / 365. - Daily Average for the year
   COUNT(ResidentKey) / 31. - Daily Average for the month of March
   COUNT(ResidentKey) / 28. - Daily Average for the month of Feb
3. Calculate Age of Resident using column Current date - ResidentDateOfBirth
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
 - FacilityName (string): Name of the Facility [Example: Parker at Somerset]
 - RecordIngestedOn (timestamp): Metadata capturing the timestamp when the record was ingested [Example: 2025-02-11 11:26:40]


TABLE: Dim_CensusLocation
TYPE: Dimension
COLUMNS:
 - LocationKey (int): Unique identifier assigned to a Location [Example: nan]
 - LocationId (string): Unique identifier assigned to a Location [Example: 1]
 - LocationCode (string): Unique code assigned to a Location [Example: 4]
 - LocationName (string): Name of the Location [Example: Somerset]
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
