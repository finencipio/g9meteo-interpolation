from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata

app = Flask(__name__)
CORS(app, origins="*", allow_headers="*", supports_credentials=True)

def extract_lon_lat_value(data: dict, keys: tuple = ("lon", "lat", "value")):
    return (np.array([x[key] for x in data]) for key in keys)


def grid(data):
    lon, lat, value = extract_lon_lat_value(data)
    data_to_interpolate = list(zip(lat, lon))

    X, Y = np.linspace(13, 20, int(7 / 0.2)), np.linspace(42, 47, int(5 / 0.2))

    X, Y = np.meshgrid(X, Y)
    #interp = LinearNDInterpolator(data_to_interpolate, value)
    img = griddata(data_to_interpolate, value, (X, Y), method='linear')#interp(X, Y)

    img_n = griddata(data_to_interpolate, value, (X, Y), method='nearest')

    img = np.where(np.isnan(img), img_n, img)

    result = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            result.append({"lon": X[i, j], "lat": Y[i, j], "value": img[i, j]})

    return result


def points(data):
    lon_src, lat_src, value_src = extract_lon_lat_value(data['src'])
    lon_dst, lat_dst = extract_lon_lat_value(data['dst'], ("lon", "lat"))

    points_src = np.vstack((lon_src, lat_src)).T
    points_dst = np.vstack((lon_dst, lat_dst)).T

    int_values = LinearNDInterpolator(points_src, value_src)(points_dst)#interpolate_to_points(points_src, value_src, points_dst, 'linear')
    int_values_2 = NearestNDInterpolator(points_src, value_src)(points_dst)

    int_values = np.where(np.isnan(int_values), int_values_2, int_values)

    result = []
    for i in range(lon_dst.shape[0]):
        result.append({"lon": lon_dst[i], "lat": lat_dst[i], "value": int_values[i]})

    return result


@app.route('/interpolate_to_points', methods=['POST'])
def interpolate_to_points():
    data = request.json
    result = points(data)
    response = jsonify(result)
    return response


@app.route('/interpolate_to_grid', methods=['POST'])
def interpolate_to_grid():
    data = request.json
    result = grid(data['src'])
    return jsonify(result)
