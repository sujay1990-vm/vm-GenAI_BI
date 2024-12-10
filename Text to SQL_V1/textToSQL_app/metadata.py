import pandas as pd

def create_table_metadata():
    metadata = [
        {"Column Name": "claim_no", "Description": "Unique identifier for the insurance claim."},
        {"Column Name": "fnol_call", "Description": "First Notice of Loss call identifier or details."},
        {"Column Name": "LossDate", "Description": "Date of the reported loss."},
        {"Column Name": "loss_cause", "Description": "Cause of the loss (e.g., fire, theft)."},
        {"Column Name": "loss_location_zip", "Description": "ZIP code where the loss occurred."},
        {"Column Name": "claim_description", "Description": "Detailed description of the claim."},
        {"Column Name": "LOB", "Description": "Line of Business associated with the claim."},
        {"Column Name": "PolicyNumber", "Description": "Policy number linked to the claim."},
        {"Column Name": "direct_paid_loss", "Description": "Amount directly paid for the loss."},
        {"Column Name": "direct_paid_LAE", "Description": "Amount directly paid for Loss Adjustment Expenses."},
        {"Column Name": "direct_outstanding_loss", "Description": "Outstanding amount for the loss."},
        {"Column Name": "CAT_flag", "Description": "Indicates if the claim is related to a catastrophe event."},
        {"Column Name": "loss_year", "Description": "Year when the loss occurred."},
        {"Column Name": "Age_max", "Description": "Maximum age of the insured person(s)."},
        {"Column Name": "CoverageA_Limit", "Description": "Limit of Coverage A for the policy."},
        {"Column Name": "ConstructionYear", "Description": "Year the insured property was constructed."},
        {"Column Name": "DriverCount", "Description": "Number of drivers covered under the policy."},
        {"Column Name": "CreditScore", "Description": "Credit score of the insured person or entity."},
        {"Column Name": "bi_limit_1", "Description": "First limit for bodily injury coverage."},
        {"Column Name": "bi_limit_2", "Description": "Second limit for bodily injury coverage."},
        {"Column Name": "one_auto_full_cov_flag", "Description": "Indicates if one auto has full coverage."},
        {"Column Name": "Age_min", "Description": "Minimum age of the insured person(s)."},
        {"Column Name": "MaritalStatus", "Description": "Marital status of the insured person."},
        {"Column Name": "Gender", "Description": "Gender of the insured person."},
        {"Column Name": "VehicleCount", "Description": "Number of vehicles covered under the policy."},
        {"Column Name": "AgencyProductNam", "Description": "Name of the agency product linked to the policy."},
        {"Column Name": "EarnedPremium", "Description": "Earned premium amount for the policy."},
        {"Column Name": "AnnualizedIPAmt", "Description": "Annualized installment premium amount."},
        {"Column Name": "minPolicyEffectiveDt", "Description": "Earliest effective date of the policy."},
        {"Column Name": "minPolicyInceptionDts", "Description": "Earliest inception date of the policy."},
        {"Column Name": "maxPolicyExpirationDt", "Description": "Latest expiration date of the policy."},
        {"Column Name": "maxPolicyCancelDt", "Description": "Latest cancellation date of the policy."},
        {"Column Name": "IncurredLossCat", "Description": "Incurred loss amount for catastrophe claims."},
        {"Column Name": "IncurredLossNonCat", "Description": "Incurred loss amount for non-catastrophe claims."},
        {"Column Name": "adjuster_notes", "Description": "Notes or comments added by the claim adjuster."},
        {"Column Name": "num_adjuster_notes", "Description": "Number of notes made by the adjuster."}
    ]
    
    # Format metadata as a string for LLM input
    formatted_metadata = "\n".join(
        f"Column Name: {col['Column Name']}, Description: {col['Description']}" for col in metadata
    )
    return formatted_metadata

# Example usage
metadata_text = create_table_metadata()


def generate_table_schema():
    schema = [
        {"Column Name": "claim_no", "Data Type": "VARCHAR", "Null": "NO", "Primary Key": "YES"},
        {"Column Name": "fnol_call", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "LossDate", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "loss_cause", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "loss_location_zip", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "claim_description", "Data Type": "TEXT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "LOB", "Data Type": "VARCHAR", "Null": "NO", "Primary Key": "NO"},
        {"Column Name": "PolicyNumber", "Data Type": "VARCHAR", "Null": "NO", "Primary Key": "NO"},
        {"Column Name": "direct_paid_loss", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "direct_paid_LAE", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "direct_outstanding_loss", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "CAT_flag", "Data Type": "BOOLEAN", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "loss_year", "Data Type": "INTEGER", "Null": "NO", "Primary Key": "NO"},
        {"Column Name": "Age_max", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "CoverageA_Limit", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "ConstructionYear", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "DriverCount", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "CreditScore", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "bi_limit_1", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "bi_limit_2", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "one_auto_full_cov_flag", "Data Type": "BOOLEAN", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "Age_min", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "MaritalStatus", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "Gender", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "VehicleCount", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "AgencyProductNam", "Data Type": "VARCHAR", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "EarnedPremium", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "AnnualizedIPAmt", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "minPolicyEffectiveDt", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "minPolicyInceptionDts", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "maxPolicyExpirationDt", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "maxPolicyCancelDt", "Data Type": "DATE", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "IncurredLossCat", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "IncurredLossNonCat", "Data Type": "FLOAT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "adjuster_notes", "Data Type": "TEXT", "Null": "YES", "Primary Key": "NO"},
        {"Column Name": "num_adjuster_notes", "Data Type": "INTEGER", "Null": "YES", "Primary Key": "NO"}
    ]
    
    # Format schema as a string for LLM input
    formatted_schema = "\n".join(
        f"Column Name: {col['Column Name']}, Data Type: {col['Data Type']}, Null: {col['Null']}, Primary Key: {col['Primary Key']}"
        for col in schema
    )
    return formatted_schema

# Example usage

