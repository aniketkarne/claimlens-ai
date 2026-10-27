"""System prompt for the qwen2.5:7b reasoning layer.

The prompt lives in its own module so the contract it defines can be
imported and tested without pulling in the HTTP client. The reasoner
imports it from here; the contract test pins its contents.
"""

QWEN25_SYSTEM_PROMPT = """Task: Extract structured data from the European Motor Accident Statement text below.

Return a valid JSON object with these exact fields (null if not found):

{
  "claim_type": "motor_accident | theft | vandalism | weather_damage | unknown",
  "fraud_indicators": ["string"] or null,
  "accident_date": "DD-MM-YYYY or null",
  "accident_time": "HH:MM or null",
  "location": "string or null",
  "injuries": true/false/null,
  "property_damage_other": true/false/null,
  "witnesses": ["name, address, phone"] or null,
  "insured_name_a": "string or null",
  "insured_address_a": "string or null",
  "vehicle_a": "make model or null",
  "plate_a": "registration or null",
  "insurance_a": "company name or null",
  "policy_a": "policy number or null",
  "driver_name_a": "string or null",
  "license_a": "license number or null",
  "point_of_impact_a": "string or null",
  "damage_a": "description or null",
  "remarks_a": "string or null",
  "insured_name_b": "string or null",
  "insured_address_b": "string or null",
  "vehicle_b": "make model or null",
  "plate_b": "registration or null",
  "insurance_b": "company name or null",
  "policy_b": "policy number or null",
  "driver_name_b": "string or null",
  "license_b": "license number or null",
  "point_of_impact_b": "string or null",
  "damage_b": "description or null",
  "remarks_b": "string or null",
  "circumstances_a": [1,2,...] or null,
  "circumstances_b": [1,2,...] or null,
  "accident_summary": "string or null",
  "fault_assessment": "string or null",
  "requires_more_info": ["string"] or null
}

RULES:
- Return ONLY valid JSON. No markdown. No explanation. No thinking.
- Extract what you see. If not present, use null.
- For witnesses: extract name, address and phone if present.
- Date format: DD-MM-YYYY
- claim_type is the category of the claim: motor_accident, theft, vandalism, weather_damage, or unknown. Use unknown only when the text gives no indication at all.
- fraud_indicators lists concrete red flags visible in the text (impossible dates, contradictory statements, signs of a staged claim). Return null when there is nothing suspicious. Do not accuse the claimant."""
