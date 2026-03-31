# Trip Stitcher

![banner.png](assets/banner.png)

## Project Overview

This project provides tools for analyzing public transport trips, with a focus on bus operations. It is designed to help researchers, planners, and developers understand trip patterns, energy requirements, and vehicle allocation.  

The project offers three main functionalities:

1. **Extract Public Bus Trips from GTFS Data**  
   Parse and process GTFS feeds to extract individual bus trips, including stop sequences, timings, and route information.

2. **Estimate Energy Demands of Trips**  
   Compute energy consumption for each extracted trip based on vehicle parameters and trip characteristics, helping evaluate operational efficiency and sustainability.

3. **Stitch Trips into Driving Missions**  
   Allocate trips to vehicles by "stitching" individual bus trips together into driving missions, supporting vehicle scheduling and fleet optimization.

## Installation

This installation assumes the use of *venv*.

**1. Clone the repository:**

```
git clone git@github.com:pinti-zh/trip-stitcher.git
cd trip-stitcher
```

**2. Create a virtual environment:**

Linux/macOS:
```
python3 -m venv venv
source venv/bin/activate
```

Windows (Powershell):
```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3. Upgrade pip:**

```
pip install --upgrade pip
```

**4. Install ocsept:**

```
pip install git+https://gitlab.com/ocsept/ocsept.git@develop
```

**5. Install project:**

```
pip install .
```

or if you also want to use the notebooks

```
pip install ".[notebooks]"
```

## Usage

**1. Activate the virtual environment**

Linux/macOS:
```
source venv/bin/activate
```
Windows (Powershell):
```
.\venv\Scripts\Activate.ps1
```

**2. [Optional] Create data and output directories:**

```
mkdir data
mkdir output
```

**3. Read the GTFS data into a database:**

This step assumes that you have a directory containing
all the .txt files of the GTFS dataset.

```
build-db --gtfs-directory <your-gtfs-directory> 
```

This can take a few minutes.

After successful execution of the script the data is stored in a database for faster access.

**4. Extract relevant information into a parquet file:**

Chose a destination for the relevant data, for example `data/postauto.parquet`.

```
build-parquet --query-output-file <your-destination> --agency postauto
```

**5. Extract trips and estimate energy demands:**

To extract the trips and calculate the energy demands we use three scripts:
1. `find-trips`: Extracts trips near a specified location.
2. `calculate-energy-demand`: Augments trips with estimated energy demands.
3. `stitch`: "Stitches" trips together into driving missions. 

All scripts output structured JSON Lines to stdout and log to stderr. This means you can specify output files or pipe the scripts to one another, and combine them with Unix commands. For example, you can run:

```
find-trips --file data/postauto.parquet | head -n 8 | stitch > output/driving_missions.jsonl
```

This will find trips near a location and stitches together the first 8 into driving missions.

**Warning:** Since `calculate-energy-demand` uses public APIs, an internet connection is required to run the script.

## Notebooks

This repository includes Jupyter notebooks designed to help you understand the key concepts behind the project. They provide step-by-step explanations and examples, showing how the code works and how it should be used effectively.  

Explore these notebooks to get a hands-on understanding and see the best practices in action.

## Future Improvements

- Add service trips from and to depots.
- Add logic for buses to return to depots based on energy consumption.
- Differentiate between separate stops and split terminals.
- Get speed limits from an API instead of guessing.

## Acknowledgements

**Main Contributors:**
- Luca Pinter
- Fabio Widmer
- Andreas Hiropedi

### Special Thanks
We would also like to thank the following organizations and individuals for their support and contributions:

- **Organizations**
  - PostAuto Switzerland
  - Zurich Information Security & Privacy Center (ZISC)
  - Institute for Dynamic Systems and Control (IDSC), ETH Zürich

- **Individuals**
  - **Eric Imstepf** - for making the collaboration with PostAuto seamless, always bringing enthusiasm and encouragement to the project.
  - **Julien Burri** - for providing valuable feedback on real-world constraints during the project collaboration with PostAuto.
  - **Anina Leuch & Lars Schmutz** - for providing data from PostAuto, offering valuable feedback, and consistently participating in our monthly meetings. 
  - **Dr. Kari Kostiainen** - for supporting the project through ZISC and making collaboration effortless by keeping formalities and bureaucracy to a minimum.
  - **Prof. Dr. Christopher Onder** – for his support as head of the IDSC research group.

## License

This project is released under the **MIT License**.
