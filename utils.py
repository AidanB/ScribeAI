is_capital = lambda x: (x>="A") and (x<="Z")

def as_numbered_list(inList):
    output = ""
    for i,x in enumerate(inList):
        output += f"{i+1}. {x}" # account for zero-indexing

    return output

def hpi_schema_to_json(inData):
    output = {"findings":[]}

    for hpics in inData:
        statement = hpics.statement
        justification = hpics.justification
        output["findings"].append({"statement":statement,"justification":justification})

    return

