# Differential Privacy

## Installation

To install all the dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

### Simple Query

To run the simple query under DP:

(epsilon = 1.0, beta = 0.1)

```
python ip_simple_query.py
```

### Full Query

To run the full query under DP:

(epsilon = 1.0, beta = 0.1)

```
python ip_full_query.py
```

## Parameter Studies

### Epsilon Study

To study the effects of different epsilon values:

```
python epsilon.py
```

### Beta Study

To study the effects of different beta values:

```
python beta.py
```

