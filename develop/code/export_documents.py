import json

with open("C:/Team-Ragoon/project-docs/develop/dataset/drug_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

output = ""
for i, doc in enumerate(documents):
    output += doc["page_content"]
    output += "\n\n" + "=" * 50 + "\n\n"

with open("C:/Team-Ragoon/project-docs/develop/dataset/drug_documents.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(f"총 {len(documents)}개 → C:/Team-Ragoon/project-docs/develop/dataset/drug_documents.txt 저장 완료")
