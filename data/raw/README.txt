Place dataset images here after downloading from:
https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view

Expected folder structure after extraction:
data/raw/
├── underripe/   ← green hard avocados
├── breaking/    ← olive/yellow-green transitioning
├── ripe/        ← dark purple/black firm avocados
└── overripe/    ← very dark, wrinkled, decaying avocados

You can use the download helper in Notebook 01:
    from src.dataset import download_dataset
    download_dataset(file_id='1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET')
