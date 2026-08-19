from ucimlrepo import fetch_ucirepo, list_available_datasets

# check which datasets can be imported
list_available_datasets()

# import dataset
connectionist_bench_sonar_mines_vs_rock = fetch_ucirepo(id=151)

# access data
X = connectionist_bench_sonar_mines_vs_rocks.data.features
y = connectionist_bench_sonar_mines_vs_rocks.data.targets

# metadata
print(connectionist_bench_sonar_mines_vs_rocks.metadata)

# variable information
print(connectionist_bench_sonar_mines_vs_rocks.variables)

# access variable info in tabular format
print(heart_disease.variables)