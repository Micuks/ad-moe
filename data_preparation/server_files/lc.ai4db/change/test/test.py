import csv

def save_dict_to_csv(data, filename):
    """
    Saves the provided dictionary data to a CSV file.

    :param data: List of dictionaries, each dictionary contains keys and values corresponding
                 to column names and their values.
    :param filename: The name of the file to save the data to.
    """
    # 确定字典中的键将作为列标题
    fieldnames = data[0].keys()

    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入列标题
        writer.writeheader()

        # 写入数据
        for entry in data:
            writer.writerow(entry)

# 示例数据
sample_data = [
    {'Time': '2023-11-27 10:00:00.000', 'Tenant': 'tenant1', 'IP': '192.168.1.1', 'Port': '8080', 'Name': 'exampleName', 'Value': 100},
    {'Time': '2023-11-27 10:00:00.000', 'Tenant': 'tenant1', 'IP': '192.168.1.1', 'Port': '8080', 'Name': 'exampleName', 'Value': 100},
    # ... 更多数据 ...
]

# 将数据保存到 CSV 文件
save_dict_to_csv(sample_data, 'output.csv')
