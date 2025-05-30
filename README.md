# RPL_docLoader

# Ingest a file
python cli.py ingest ./docs/test.pdf

# Query it
python cli.py query "What did the experiment conclude about permittivity?"




# Detect intent
python cli.py intent "create a project called quantum sensors"

# Ingest a file
python cli.py ingest notes_day2.pdf --project quantum_sensors

# Ask a question
python cli.py query "What did we find about sapphire?" --project quantum_sensors