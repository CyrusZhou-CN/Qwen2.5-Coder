# Top Tech Startups Hiring 2023

<br>

## About
<br>

This dataset contains data on top tech startups and their hiring information. Some of the listed features are company name, description of the company, company's website URL, industries in which the company operates, hiring activity of the company, job vacancies etc. Please refer to `data_dictionary.txt` for a complete list of features.

This dataset also has images of the company logo. These images are present in the `./images` folder and are named with respect to their company `id`. The maximum dimensions of each image are 200 x 200 pixels, while some images have rectangular shapes having value below 200 pixels.

The dataset is provided in 2 formats i.e., `json` and `csv`. As some features have array and key-value pair as their datatypes, the suitable file format was json. But the scientific community is more inclined towards tabular format hence; csv file format is also provided. Note while transforming data to csv format, array and key-value datatypes were slightly modified. Please refer to the schema files to get a feeling for the structure of these datatypes.

A quick note about missing values. This dataset contains some missing values, they are represented using empty strings `""`. When working with the data, please take appropriate measures.

At last, most of the irregularities from the dataset were handled, and if you find any, please report it, and it will be fixed in the next version.

<br>

**Note:**
The dataset is for educational use only. The data owners are the respective companies and the websites providing the data.
That being said, the jobs listed in the dataset are real, and if interested, you can apply to them from the respective company's website.

<br>
<br>

## Content

<br>

`images/`: &emsp; images of company logo
`json_schema.txt`: &emsp; schema for json dataset
`csv_schema.txt`: &emsp; schema for csv dataset
`data_dictionary.txt`: &emsp; data dictionary for the dataset
`json_data.json`: &emsp; json dataset
`csv_data.csv`: &emsp; csv dataset
`README.md`: &emsp; readme file