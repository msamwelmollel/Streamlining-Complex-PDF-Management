# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 08:52:17 2023

@author: msamwelmollel
"""

from llama_index.llama_pack import download_llama_pack
from embedded_tables_unstructured_pack.base import EmbeddedTablesUnstructuredRetrieverPack
import subprocess
import csv



# #Remove the comment the first time, and once the pack has been downloaded, you can then add the comment back.
# #*****************
# EmbeddedTablesUnstructuredRetrieverPack = download_llama_pack(
#     "EmbeddedTablesUnstructuredRetrieverPack",
#     "./embedded_tables_unstructured_pack",
# )
# #*************



def convert_pdf_to_html(pdf_path, html_path):
    command = f"pdf2htmlEX {pdf_path} --dest-dir {html_path}"
    subprocess.call(command, shell=True)

input_pdf = "quarterly-nvidia.pdf"
output_pdf = "quarterly-nvidia"

convert_pdf_to_html(input_pdf, output_pdf)






# from embedded_tables_unstructured_pack.base import EmbeddedTablesUnstructuredRetrieverPack

embedded_tables_unstructured_pack = EmbeddedTablesUnstructuredRetrieverPack(
    "quarterly-nvidia/quarterly-nvidia.html",
    nodes_save_path="nvidia-quarterly.pkl"
)



# Function to read questions from the CSV file
def read_questions(filename):
    questions = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            questions.append(row[0])  # Assuming questions are in the first column
    return questions

# Function to write responses to the CSV file
def write_response(filename, question, response):
    with open(filename, 'a', newline='') as csvfile:  # Open in append mode
        writer = csv.writer(csvfile)
        writer.writerow([question, response])

# Main loop
questions = read_questions('questions.csv')

while True:
    # Load questions from the CSV file (e.g., 'questions.csv')
    

    # If there are no more questions, exit the loop
    if not questions:
        print("No more questions in the CSV file.")
        break

    # Get the next question
    query = questions.pop(0)

    # Process the query and get the response
    response = embedded_tables_unstructured_pack.run(query)

    # Print and store the response in the CSV file
    print("*********")
    print('Question: ', query)
    print('Response: ', str(response))

    # Write the response to the CSV file (e.g., 'responses.csv')
    write_response('responses.csv', query+str(response), str(response))




