
census_domain_instructions = """
Domain Specific Instructions:
**Always** use the right date format : Example : CensusDateDate [Example: 1961-04-14 00:00:00]
1. **Definition of Census**: 
   - "Census" refers to the total number of residents.
   - Always compute this using: `COUNT(ResidentKey)` , **DO NOT** use `COUNT(DISTINCT ResidentKey)` as it may lead to incorrect results.
   - Never use `AVG(ResidentKey)` as ResidentKey is just a surrogate key, not a measure.

2. **Average Census by Day of Week**:
   - To calculate average census per day of the week (e.g., average Monday census vs. average Tuesday census), group by the day name and apply `AVG()` directly on the daily counts.
   - Do **not** divide total count by number of days in the month or year

3. **Age of Resident**:
   - When asked for age, calculate: `CurrentDate - ResidentDateOfBirth`
   - Use appropriate date difference function depending on SQL dialect (e.g., `DATEDIFF`, `YEAR(CURRENT_DATE) - YEAR(...)`, etc.)

4. **MTD (Month-to-Date)**:
   - When asked for Month-to-Date, include all dates from the **1st of the month up to the specified date**.

5. **YTD (Year-to-Date)**:
   - When asked for Year-to-Date, include all dates from **January 1st of the year up to the specified date**.

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
5. Can you provide a chart showing the Census data for March 1st, 2024, in Somerset, broken down by unit name?
Answer :
SELECT 
    u.UnitName AS UnitName, 
    COUNT(f.ResidentKey) AS DailyCensus
FROM 
    Fact_Census f
JOIN 
    Dim_CensusDate d ON f.CensusDateKey = d.CensusDateKey
JOIN 
    Dim_CensusLocation l ON f.LocationKey = l.LocationKey
JOIN 
    Dim_CensusUnit u ON f.UnitKey = u.UnitKey
WHERE 
    d.CensusDateDate = '2024-03-01 00:00:00'
    AND l.LocationName = 'Somerset'
GROUP BY 
    u.UnitName;
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