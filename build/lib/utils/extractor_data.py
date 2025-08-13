import pandas as pd
import re
import json
import csv
from collections import defaultdict



filename = "X13.xlsx"
filter = "X13"


df = pd.read_excel(
    filename,
    sheet_name="Motherboard Reference",   
    engine="openpyxl"
)

 
def get_str_or_empty(row, key):
    val = row.get(key, "")
    if pd.isna(val):
        return ""
    return str(val).strip()

def get_int_or_empty(row, column_name):
    try:
        val = row[column_name]
        if pd.notna(val):
            return int(val)
    except (ValueError, TypeError, KeyError):
        pass
    return ""
 
filtered_df = df[df["Model"].astype(str).str.contains(filter, case=False, na=False)]

results_mb_spec = []

for model, group in filtered_df.groupby("Model"):
    row = group.iloc[0]   


    only_superserver_raw = get_str_or_empty(row, "For Superserver Only").strip().lower()

    if only_superserver_raw == "yes":
        only_superserver_value = "Available only as part of a complete server system"
    elif only_superserver_raw == "no":
        only_superserver_value = "Available either as a standalone motherboard or within a complete server system"
    else:
        only_superserver_value = ""


    mb_gen = {
        "generation": get_str_or_empty(row, "Generation")
    },
    family_info ={
        "product_family": get_str_or_empty(row, "Product Family")
    },
    motherboard_type ={
        "product_type":get_str_or_empty(row, "Category")

    },
    spec_info = {
        "description": get_str_or_empty(row, "Description"),
        "only_superserver": only_superserver_value,
        "cpu_tdp": f"{get_str_or_empty(row, "CPU TDP Support")}",

        "operating_environment": {
            "operating_temperature": (
                f'{get_int_or_empty(row, "Min Operating Temperature (C)")}°C to '
                f'{get_int_or_empty(row, "Max Operating Temperature (C)")}°C '
                f'({get_int_or_empty(row, "Min Operating Temperature (F)")}°F to '
                f'{get_int_or_empty(row, "Max Operating Temperature (F)")}°F)'
            ),
            "non-operating_temperature": (
                f'{get_int_or_empty(row, "Min Non-Operating Temperature (C)")}°C to '
                f'{get_int_or_empty(row, "Max Non-Operating Temperature (C)")}°C '
                f'({get_int_or_empty(row, "Min Non-Operating Temperature (F)")}°F to '
                f'{get_int_or_empty(row, "Max Non-Operating Temperature (F)")}°F)'
            ),               
            "operating_relative_humidity": (
                f'{get_int_or_empty(row, "Min Operating Relative Humidity P")}% to '
                f'{get_int_or_empty(row, "Max Operating Relative Humidity P")}% '
                f'({get_str_or_empty(row, "Operating Relative Humidity Type")})'
            ),
            "non-operating_relative_humidity": (
                f'{get_int_or_empty(row, "Min Non-Operating Relative Humidity P")}% to '
                f'{get_int_or_empty(row, "Max Non-Operating Relative Humidity P")}% '
                f'({get_str_or_empty(row, "Non-Operating Relative Humidity Type")})'
            ),
        },
        "pm_name": get_str_or_empty(row, "PM Name"),
        "upi_count": get_int_or_empty(row, "upiCount"),
        "category": get_str_or_empty(row, "Category"),
        "memory_size": get_str_or_empty(row, "DIMM Sizes"),
        "memory_speed": get_str_or_empty(row, "Memory Speed (MHz)"),
        "fan_header_count": get_int_or_empty(row, "fan Header Count"),
        "fan_header_type": get_str_or_empty(row, "fan Header Type"),
        "is_bmc_fw": get_str_or_empty(row, "has BMC firmware"),
        "is_oob": get_str_or_empty(row, "OOB"),
        "form_factor": get_str_or_empty(row, "form Factor"),
        "dimension": f'{get_str_or_empty(row, "width (Inches)")}" x {get_str_or_empty(row, "height (Inches)")}" ({get_str_or_empty(row, "width (Cm)")}cm x {get_str_or_empty(row, "height (Cm)")}cm)',
        "error_detection": get_str_or_empty(row, "Error Detection"),
        "has_bmc_firmware": get_str_or_empty(row, "has BMC firmware")
    }

    results_mb_spec.append({
        "motherboard_name": model,
        "motherboard_generation": mb_gen,
        "motherboard_general_spec": spec_info,
        "product_family": family_info,
        "motherboard_type": motherboard_type
    })

 
print(json.dumps(results_mb_spec, indent=2, ensure_ascii=False))


 
df = pd.read_excel(
    filename,
    sheet_name="Health Monitoring",
    usecols=["MB Model", "category", "name"],
    engine="openpyxl"
)

 
def clean_text(text):
    text = re.sub(r"<br\s*/?>", "\n", str(text))   
    text = re.sub(r"[®™©]", "", text)             
    return text.strip()

 
filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

 
results_health_monitoring = []

for mb_model, group in filtered_df.groupby("MB Model"):
    features = {}

    for _, row in group.iterrows():
        f_type = row["category"]
        feature_text = clean_text(row["name"])
        
 
        if f_type in features:
            features[f_type] += ", " + feature_text
        else:
            features[f_type] = feature_text

    results_health_monitoring.append({
        "motherboard_name": mb_model,
        "health_monitoring": features
    })
 
print(json.dumps(results_health_monitoring, indent=2, ensure_ascii=False))






df = pd.read_excel(
    filename,
    sheet_name="BIOS",   
    usecols=["MB Model", "type", "brand", "romSizeMB", "features"],
    engine="openpyxl"
)
 
def get_str_or_empty(row, key):
    val = row.get(key, "")
    if pd.isna(val):
        return ""
    return str(val).strip()
 
filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

results_bios = []

for mb_model, group in filtered_df.groupby("MB Model"):
    row = group.iloc[0]   
    bios_info = {
        "type": get_str_or_empty(row, "type"),
        "brand": get_str_or_empty(row, "brand"),
        "romSizeMB": get_str_or_empty(row, "romSizeMB"),
        "features": get_str_or_empty(row, "features")
    }
    results_bios.append({
        "motherboard_name": mb_model,
        "bios": bios_info
    })

 
print(json.dumps(results_bios, indent=2, ensure_ascii=False))




df = pd.read_excel(
    filename,
    sheet_name="Input Output",
    usecols=[
        "MB Model", "port_type", "num_of_ports", "interface", "speed_gbps",
        "interface_version", "controller_type", "controller_location", "feature", "grouping", "note"
    ],
    engine="openpyxl"
)

def clean_text(text):
    text = re.sub(r"<br\s*/?>", "\n", str(text))   
    text = re.sub(r"[®™©]", "", text)  
    return text.strip()
 
filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

results_interfaces = []

for mb_model, group in filtered_df.groupby("MB Model"):
    features = {}

 

    for _, row in group.iterrows():
        f_type = str(row["interface"]).strip()   
        
 
        desc_dict = {}

        if pd.notna(row.get("port_type")):
            desc_dict["port_type"] = str(row["port_type"]).strip()
        if pd.notna(row.get("num_of_ports")):
            desc_dict["count"] = str(int(row["num_of_ports"]))
        if pd.notna(row.get("speed_gbps")):
            desc_dict["speed"] = f"{row['speed_gbps']}Gbps"
        if pd.notna(row.get("interface_version")):
            desc_dict["interface_version"] = str(row["interface_version"])
        if pd.notna(row.get("controller_type")):
            desc_dict["controller_type"] = str(row["controller_type"])
        if pd.notna(row.get("controller_location")):
            desc_dict["controller_location"] = str(row["controller_location"])
        if pd.notna(row.get("feature")):
            desc_dict["feature"] = str(row["feature"])
        if pd.notna(row.get("note")):
            desc_dict["note"] = str(row["note"])
        
         
        if f_type in features:
             
            if isinstance(features[f_type], list):
                features[f_type].append(desc_dict)
            else:
                 
                features[f_type] = [features[f_type], desc_dict]
        else:
            features[f_type] = desc_dict

    results_interfaces.append({
        "motherboard_name": mb_model,
        "interfaace": features
    })

 
print(json.dumps(results_interfaces, indent=2, ensure_ascii=False))






df = pd.read_excel(
    filename,
    sheet_name="Key Features",
    usecols=[
        "MB Model", "keyFeatures", "type"
    ],
    engine="openpyxl"
)

def clean_text(text):
    text = re.sub(r"<br\s*/?>", "\n", str(text))   
    text = re.sub(r"[®™©]", "", text)   
    return text.strip()

filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

results_key_features = []

for mb_model, group in filtered_df.groupby("MB Model"):
    features = {}

    for _, row in group.iterrows():
        f_type = row["type"]
        feature_text = clean_text(row["keyFeatures"])
        if f_type in features:
            features[f_type] += ", " + feature_text   
        else:
            features[f_type] = feature_text

    results_key_features.append({
        "motherboard_name": mb_model,
        "key_features": features
    })

 
print(json.dumps(results_key_features, indent=2, ensure_ascii=False))




 

df = pd.read_excel(
    filename,
    sheet_name="Expansion Slots",
    usecols=[
        "MB Model", "type", "name", "count", "description", "isOptional"
    ],
    engine="openpyxl"
)


def to_int_or_empty(val):
    try:
        return int(val)
    except:
        return ""

def get_str_or_empty(row, key):
    val = row.get(key, "")
    if pd.isna(val):
        return ""
    return str(val).strip()

 
filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

results_expansion_slots = []

for mb_model, group in filtered_df.groupby("MB Model"):
    pcie_ports = []
    for _, row in group.iterrows():
        port_info = {
            "type": get_str_or_empty(row, "type"),
            "name": get_str_or_empty(row, "name"),
            "count": to_int_or_empty(row.get("count", "")),
            "description": get_str_or_empty(row, "description"),
            "is_optional": get_str_or_empty(row, "isOptional"),
        }
        pcie_ports.append(port_info)

    results_expansion_slots.append({
        "motherboard_name": mb_model,
        "expansion_slots": pcie_ports
    })

print(json.dumps(results_expansion_slots, indent=2, ensure_ascii=False))


 



df = pd.read_excel(
    filename,
    sheet_name="M.2 Interface",
    usecols=[
        "MB Model", "name", "m2Key", "formFactorLength", "count", "raidSupport", "description"
    ],
    engine="openpyxl"
)


def to_int_or_empty(val):
    try:
        return int(val)
    except:
        return ""

 
def get_str_or_empty(row, key):
    val = row.get(key, "")
    if pd.isna(val):
        return ""
    return str(val).strip()

 
filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

results_m2_interface = []

for mb_model, group in filtered_df.groupby("MB Model"):
    parts = []
    for _, row in group.iterrows():
        part_info = {
            "name": get_str_or_empty(row, "name"),
            "m2_key_type": get_str_or_empty(row, "m2Key"),
            "form_factor_length": get_str_or_empty(row, "formFactorLength"),
            "count": to_int_or_empty(row.get("count", "")),
            "raidSupport": get_str_or_empty(row, "raidSupport"),
            "description": get_str_or_empty(row, "description"),
        }
        parts.append(part_info)

    results_m2_interface.append({
        "motherboard_name": mb_model,
        "m.2_interface": parts
    })

print(json.dumps(results_m2_interface, indent=2, ensure_ascii=False))


df = pd.read_excel(
    filename,
    sheet_name="Storage",
    usecols=[
        "MB Model", "interface_name", "data_rate_gbps", "port_type", "num_of_ports", "num_of_drives", "via_expander", "controller_name", "controller_type", "controller_location", "raid_support","note" 
    ],
    engine="openpyxl"
)



def to_int_or_empty(val):
    try:
        return int(val)
    except:
        return ""

def get_str_or_empty(row, key):
    val = row.get(key, "")
    if pd.isna(val):
        return ""
    return val


filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

results_storage_specs = []

for mb_model, group in filtered_df.groupby("MB Model"):
    interfaces = []
    for _, row in group.iterrows():
        interface_info = {
            "interface_name": get_str_or_empty(row, "interface_name"),
            "data_rate_gbps": to_int_or_empty(row.get("data_rate_gbps", "")),
            "port_type": get_str_or_empty(row, "port_type"),
            "num_of_ports": to_int_or_empty(row.get("num_of_ports", "")),
            "num_of_drives": to_int_or_empty(row.get("num_of_drives", "")),
            "via_expander": get_str_or_empty(row, "via_expander"),
            "controller_name": get_str_or_empty(row, "controller_name"),
            "controller_type": get_str_or_empty(row, "controller_type"),
            "controller_location": get_str_or_empty(row, "controller_location"),
            "raid_support": get_str_or_empty(row, "raid_support"),
            "note": get_str_or_empty(row, "note"),
        }
        interfaces.append(interface_info)

    results_storage_specs.append({
        "motherboard_name": mb_model,
        "storage_specs": interfaces
    })

print(json.dumps(results_storage_specs, indent=2, ensure_ascii=False))



df = pd.read_excel(
    filename,
    sheet_name="System Memory",
    usecols=[
        "MB Model", "Interface", "type", "maxCapacity", "ECC", "Buffered", "maxSpeed", "sizesGb","media"
    ],
    engine="openpyxl"
)


filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

results_memory_spec = []
for _, row in filtered_df.iterrows():
    sizes_raw = row.get("sizesGb", "")
    if pd.isna(sizes_raw) or sizes_raw == "":
        sizes_list = []
    else:
        sizes_list = [int(s.strip()) for s in str(sizes_raw).split(",") if s.strip().isdigit()]

    memory_spec = {
        "memory_type": f"{row["Interface"]} {row["type"]}",
        "max_capacity": row["maxCapacity"],
        "error_correction": row["ECC"],
        "media_type": row["media"],
        "buffering": row["Buffered"],
        "max_speed_mtps": row["maxSpeed"],
        "max_module_size": row["sizesGb"]
    }

    results_memory_spec.append({
        "motherboard_name": row["MB Model"],
        "memory_specs": memory_spec
    })

import json
print(json.dumps(results_memory_spec, indent=2, ensure_ascii=False))



df = pd.read_excel(
    filename,
    sheet_name="Optional Parts",
    usecols=[
        "MB Model", "Name", "partNumber", "quantity", "description", "partsType"
    ],
    engine="openpyxl"
)


filtered_df = df[df["MB Model"].str.contains(filter, case=False, na=False)]

 
results_optional_parts = []

for mb_model, group in filtered_df.groupby("MB Model"):
    parts = []
    for _, row in group.iterrows():
        quantity_str = str(row["quantity"]).strip()
        try:
            quantity = int(quantity_str)
        except ValueError:
            quantity = 0  
        part = {
            "category": str(row["partsType"]).strip(),
            "name": str(row["Name"]).strip(),
            "part_number": str(row["partNumber"]).strip(),
            "quantity": quantity,
            "description": str(row["description"]).strip() if pd.notna(row["description"]) else "No description provided",
        }
        parts.append(part)

    results_optional_parts.append({
        "motherboard_name": str(mb_model).strip(),
        "optional_parts": parts
    })

print(json.dumps(results_optional_parts, indent=2, ensure_ascii=False))


 
df = pd.read_excel(
    filename,
    sheet_name="CPU",
    usecols=[
        "Generation", "SKU", "TDP (w)", "Core/Thread", "CPU Web Page Description", "Manufacturer", "Socket Supported", "Socket Alternative Name"
    ],
    engine="openpyxl"
)

filtered_df = df[df["SKU"].str.contains(filter, case=False, na=False)]



result_cpu_spec = []
for _, row in filtered_df.iterrows():
 
    if pd.isna(row["SKU"]) or filter not in row["SKU"].upper():
        continue

    motherboard_name = row["SKU"]
    cpu_spec = {
        "tdp": f'{row["TDP (w)"]}W' if pd.notna(row["TDP (w)"]) else "",
        "core_thread": row["Core/Thread"] if pd.notna(row["Core/Thread"]) else "",
        "type": re.sub(r"[®™©]", "", row["CPU Web Page Description"]) if pd.notna(row["CPU Web Page Description"]) else "",
        "Manufacturer": row["Manufacturer"] if pd.notna(row["Manufacturer"]) else "",
        "socket": (
            f'Dual Socket {row["Socket Supported"]} ({row["Socket Alternative Name"]})'
            if pd.notna(row["Socket Supported"]) or pd.notna(row["Socket Alternative Name"])
            else ""
        )
    }

    result_cpu_spec.append({
        "motherboard_name": motherboard_name,
        "cpu_spec": cpu_spec
    })


for i, item in enumerate(result_cpu_spec):
    print(f"\n=== Record {i+1} ===")
    print(json.dumps(item, indent=2, ensure_ascii=False))

 


merged_data = defaultdict(lambda: {
    "product_family": None,
    "motherboard_generation":None,
    "product_family":None,
    "motherboard_type":None,
    "motherboard_general_spec": None,
    "cpu_spec": None,
    "memory_specs": None,
    "storage_specs": None,
    "m2_interface": [],
    "expansion_slots": [],
    "optional_parts": [],
    "key_features": {},
    "interface": {},
    "bios": None,
    "health_monitoring": {}
})

for item in results_mb_spec:
    mb = item["motherboard_name"]
    merged_data[mb]["motherboard_general_spec"] = item["motherboard_general_spec"]
    merged_data[mb]["product_family"] = item["product_family"]  
    merged_data[mb]["motherboard_generation"] = item["motherboard_generation"] 
    merged_data[mb]["motherboard_type"] = item["motherboard_type"] 

for item in result_cpu_spec:
    mb = item["motherboard_name"]
    merged_data[mb]["cpu_spec"] = item["cpu_spec"]

for item in results_memory_spec:
    mb = item["motherboard_name"]
    merged_data[mb]["memory_specs"] = item["memory_specs"]

for item in results_storage_specs:
    mb = item["motherboard_name"]
    merged_data[mb]["storage_specs"] = item["storage_specs"]

for item in results_m2_interface:
    mb = item["motherboard_name"]
    merged_data[mb]["m2_interface"].extend(item["m.2_interface"])

for item in results_expansion_slots:
    mb = item["motherboard_name"]
    merged_data[mb]["expansion_slots"].extend(item["expansion_slots"])

for item in results_optional_parts:
    mb = item["motherboard_name"]
    merged_data[mb]["optional_parts"].extend(item["optional_parts"])

for item in results_key_features:
    mb = item["motherboard_name"]
    merged_data[mb]["key_features"] = item["key_features"]

for item in results_interfaces:
    mb = item["motherboard_name"]
    merged_data[mb]["interface"] = item["interfaace"]  

for item in results_bios:
    mb = item["motherboard_name"]
    merged_data[mb]["bios"] = item["bios"]

for item in results_health_monitoring:
    mb = item["motherboard_name"]
    merged_data[mb]["health_monitoring"] = item["health_monitoring"]


with open(f"merged_motherboards_{filename}.csv", "w", newline='', encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "motherboard_name",
        "motherboard_generation",
        "product_family",
        "motherboard_type",
        "motherboard_general_spec",
        "cpu_spec",
        "memory_specs",
        "storage_specs",
        "m2_interface",
        "expansion_slots",
        "optional_parts",
        "key_features",
        "interface",
        "bios",
        "health_monitoring"
    ])
    writer.writeheader()

    for mb_name, content in merged_data.items():
        writer.writerow({
            "motherboard_name": mb_name,
            "motherboard_generation": json.dumps(content["motherboard_generation"], ensure_ascii=False) if content["motherboard_generation"] else "{}",
            "product_family": json.dumps(content["product_family"], ensure_ascii=False) if content["product_family"] else "{}",
            "motherboard_type": json.dumps(content["motherboard_type"], ensure_ascii=False) if content["motherboard_type"] else "{}",
            "motherboard_general_spec": json.dumps(content["motherboard_general_spec"], ensure_ascii=False) if content["motherboard_general_spec"] else "{}",
            "cpu_spec": json.dumps(content["cpu_spec"], ensure_ascii=False) if content["cpu_spec"] else "{}",
            "memory_specs": json.dumps(content["memory_specs"], ensure_ascii=False) if content["memory_specs"] else "{}",
            "storage_specs": json.dumps(content["storage_specs"], ensure_ascii=False) if content["storage_specs"] else "[]",
            "m2_interface": ", ".join(json.dumps(p, ensure_ascii=False) for p in content["m2_interface"]),
            "expansion_slots": ", ".join(json.dumps(p, ensure_ascii=False) for p in content["expansion_slots"]),
            "optional_parts": ", ".join(json.dumps(p, ensure_ascii=False) for p in content["optional_parts"]),
            "key_features": json.dumps(content["key_features"], ensure_ascii=False) if content["key_features"] else "{}",
            "interface": json.dumps(content["interface"], ensure_ascii=False) if content["interface"] else "{}",
            "bios": json.dumps(content["bios"], ensure_ascii=False) if content["bios"] else "{}",
            "health_monitoring": json.dumps(content["health_monitoring"], ensure_ascii=False) if content["health_monitoring"] else "{}"
        })