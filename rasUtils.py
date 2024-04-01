import numpy as np

def pad_numbers(numbers):
    """
    Pad numbers with spaces to make each number string 8 characters wide. 
    This is required to overwrite HEC-RAS text geometry files, which require
    all river stations and Manning's n to be exactly 8 characters long.

    Parameters:
    - numbers (list): List of numbers to be padded.

    Returns:
    - list: List of padded number strings.
    """
    padded_numbers = []
    for num in numbers:
        padded_num = '{:>8}'.format(str(num))
        padded_numbers.append(padded_num)
    return padded_numbers

def print_padded_lines(numbers):
    """
    Print padded lines of numbers with 9 numbers per line. This is required formatting 
    for HEC-RAS' geometry files.

    Parameters:
    - numbers (list): List of numbers to be printed.

    Returns:
    - list: List of strings representing padded lines of numbers.
    """
    lines = []
    padded_numbers = pad_numbers(numbers)
    for i in range(0, len(padded_numbers), 9):
        lines.append("".join(padded_numbers[i:i+9]))
    
    lines = [line + "\n" for line in lines]
    
    return lines

def create_Manning_lines(region_station, region_mann_n):
    """
    Create Manning lines for a region.

    Parameters:
    - region_station (list): List of station values.
    - region_mann_n (list): List of Manning's n values.

    Returns:
    - list: List of strings representing Manning lines.
    """

    # ROUND TO # DECIMAL PLACES
    region_station  = [round(a, 3) for a in region_station]
    region_mann_n  = [round(a, 3) for a in region_mann_n]

    # SORT FROM CLOSEST TO FURTHEST
    sorted_list    = sorted(zip(region_station, region_mann_n, 0 * np.array(region_mann_n)))    
    flattened_list = [item for sublist in sorted_list for item in sublist]
    
    # INITIAL LINE FOR XS WRITING
    lines = [f"#Mann= {len(region_station)} ,-1,0\n"]

    # ADD ALL LINES AND RETURN
    out_l = print_padded_lines(flattened_list)
    lines.extend(out_l)
    return lines

def getElevations(idx, inf, elevs):
    li = int(inf[idx, 0])
    hi = int(inf[idx, 0]+inf[idx, 1])
    return elevs[li:hi]

def getMannings(idx, inf, mann):
    li = int(inf[idx, 0])
    hi = int(inf[idx, 0]+inf[idx, 1])
    return mann[li:hi]

def transform(x):
    return np.hstack((x.reshape(-1, 1), np.zeros((len(x), 1))))

def title(idx, atty):
    return f'River: {atty.loc[idx]["River"]} RS: {atty.loc[idx]["RS"]}'

def calcWettedPerimeter(XZ):
    # ASSUME LAST MANNING'S HAS 0 HYDRAULIC RADIUS
    # TODO: READ TOTAL CROSS SECTION LENGTH
    XZ = np.append(XZ, XZ[-1]).reshape((-1, 2))
    return np.sqrt((XZ[:, 0][1:] - XZ[:, 0][:-1]) ** 2 + (XZ[:, 1][1:] - XZ[:, 1][:-1]) ** 2)

def composeManningsN(perimeters, mannings, method="Horton"):
    if method=="Horton":
        # EQUATION 6-17 FROM CHOW
        return (np.sum(perimeters * mannings ** (1.5) / np.sum(perimeters)) ** (2/3))
    elif method=="Pavlovskii":
        return (np.sum(perimeters * mannings ** (2) / np.sum(perimeters)) ** (1/2))
    else:
        print("Compositing method not implemented, running with Horton's")
        return composeManningsN(perimeters, mannings)