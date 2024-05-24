#build the knowledge graph from csv files
import csv
import json
import os

def parse_kg_csv(file_path, file_name):
    knowledge_graph = {}
    full_path = os.path.join(file_path, file_name + '.csv')
    with open(full_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Initialize current_disease and current_attribute
            current_disease = None
            current_attribute = None
            for i, cell in enumerate(row):
                if cell:  # Check if the cell is not empty
                    if i == 0:  # The first cell has disease, attribute, and value
                        disease, attribute, value = cell.split('#')
                        current_disease = disease
                        current_attribute = attribute
                        if disease not in knowledge_graph:
                            knowledge_graph[disease] = {}
                        if attribute not in knowledge_graph[disease]:
                            knowledge_graph[disease][attribute] = []
                        knowledge_graph[disease][attribute].append(value)
                    else:  # Subsequent cells have only the value
                        knowledge_graph[current_disease][current_attribute].append(cell)

    return knowledge_graph

if __name__ == '__main__':
    file_path = 'datasets/slake/KG/'
    file_name_list = ['en_disease', 'en_organ', 'en_organ_rel']
    for file_name in file_name_list:
        print(f'Parsing {file_name}...')
        knowledge_graph = parse_kg_csv(file_path, file_name)
        with open(os.path.join(file_path, file_name + '.json'), 'w') as json_file:
            json.dump(knowledge_graph, json_file, indent=4)
