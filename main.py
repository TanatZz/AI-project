#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Songkhla routing with time-windowed roads + ML speed factor

- หา 2 เส้นทางที่ดีที่สุด (หรือมากกว่า) จาก A ไป B
- ห้ามใช้ถนนที่ "ปิด" ณ เวลานั้น (ไม่รอ)
- ใช้ ML (จำลอง) เพื่อปรับเวลาเดินทางตาม "ชั่วโมง" และ "ประเภทถนน"
"""

import sys
import os
import argparse
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#from osmnx.utils_graph import route_to_geometry  # บางเวอร์ชันต้อง import แบบนี้


# ---------- Utils ----------

def normalize_highway(value):
    """OSM 'highway' อาจเป็น list; คืนค่า str เดี่ยวๆ"""
    if isinstance(value, list):
        value = value[0]
    return str(value)

def load_graph(place):
    """โหลดกราฟถนนจาก OSM (ชนิด drive) แล้วเติมความเร็ว/เวลา"""
    ox.settings.log_console = True
    ox.settings.use_cache = True
    G = ox.graph_from_place(place, network_type="drive")
    G = ox.add_edge_speeds(G)          # เพิ่ม speed_kph (เดา/ดึงจาก OSM)
    G = ox.add_edge_travel_times(G)    # เพิ่ม travel_time (วินาที)
    return G

def build_ml_model(G, random_state=42):
    """เทรนโมเดล ML แบบจำลอง เพื่อทำนาย speed_factor(hour, highway)"""
    rng = np.random.default_rng(random_state)
    # highway types ที่พบในกราฟ
    hwy_types = set()
    for _, _, _, data in G.edges(keys=True, data=True):
        hwy_types.add(normalize_highway(data.get("highway", "unclassified")))
    hwy_types = list(hwy_types) if hwy_types else ["unclassified"]

    rows = []
    for hwy in hwy_types:
        base = {
            'motorway': 1.15, 'trunk': 1.10, 'primary': 1.00, 'secondary': 0.95,
            'tertiary': 0.95, 'residential': 0.90, 'unclassified': 0.90,
            'service': 0.90
        }.get(hwy, 0.95)
        for hour in range(24):
            # Rush hours: 7-9, 16-19 ช้าลง โดยถนนใหญ่ช้ากว่า
            if hour in [7,8,9,16,17,18,19]:
                slowdown = {'motorway': .75, 'trunk': .80, 'primary': .85,
                            'secondary': .90, 'tertiary': .92, 'residential': .95}.get(hwy, .90)
                factor = base * slowdown
            elif hour in [0,1,2,3,4,5,22,23]:
                factor = base * 1.15  # ดึกๆ โล่ง
            else:
                factor = base * 1.00

            # เพิ่ม noise เล็กน้อย
            factor = factor * (0.95 + 0.10 * rng.random())
            # ครอบช่วงที่สมเหตุสมผล
            factor = max(0.5, min(1.3, factor))
            rows.append({'hour': hour, 'highway': hwy, 'speed_factor': factor})

    df = pd.DataFrame(rows)
    X = df[['hour', 'highway']]
    y = df['speed_factor']

    pre = ColumnTransformer([
        ('highway', OneHotEncoder(handle_unknown='ignore'), ['highway']),
        ('pass', 'passthrough', ['hour'])
    ])
    model = Pipeline(steps=[
        ('preprocess', pre),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=random_state))
    ])
    model.fit(X, y)
    return model

def predict_speed_factor(model, hour, highway):
    """พยากรณ์ speed_factor จากชั่วโมงและประเภทถนน"""
    highway = normalize_highway(highway)
    X = pd.DataFrame([{'hour': int(hour) % 24, 'highway': highway}])
    f = float(model.predict(X)[0])
    return max(0.4, min(1.4, f))

def parse_time_str(s):
    return datetime.strptime(s, "%H:%M").time()

def parse_schedule(csv_path):
    """
    อ่านไฟล์เวลาเปิด–ปิดถนน
    คอลัมน์: road_name, weekday(0-6 หรือ 'all'), open_start(HH:MM), open_end(HH:MM)
    คืนค่า: dict[name_lower] = list of (wd or 'all', start_time, end_time)
    """
    df = pd.read_csv(csv_path)
    intervals = {}
    for _, row in df.iterrows():
        name = str(row['road_name']).strip().lower()
        wd_raw = str(row['weekday']).strip().lower()
        wd = 'all' if wd_raw == 'all' else int(wd_raw)
        start = parse_time_str(row['open_start'])
        end = parse_time_str(row['open_end'])
        intervals.setdefault(name, []).append((wd, start, end))
    return intervals

def is_open(name, dt_local, intervals_by_name):
    """
    ตรวจว่าถนน (ตามชื่อ name) เปิด ณ dt_local หรือไม่
    - ไม่มีชื่อ หรือไม่มีในตาราง => ถือว่าเปิด
    - รองรับช่วงเวลาข้ามเที่ยงคืน (start > end)
    """
    if not name:
        return True
    key = str(name).strip().lower()
    if key not in intervals_by_name:
        return True

    day = dt_local.weekday()
    t = dt_local.time()
    for wd, start, end in intervals_by_name[key]:
        if wd != 'all' and wd != day:
            continue
        if start <= end:
            if start <= t <= end:
                return True
        else:
            # overnight window
            if t >= start or t <= end:
                return True
    return False

import time

def build_open_weighted_graph(G, dt_local, intervals_by_name, model):
    """ลบ edge ที่ปิดออก + คำนวณ weight แบบ dynamic (แสดง progress)"""
    G2 = G.copy()
    hour = dt_local.hour
    total = G2.number_of_edges()
    start = time.perf_counter()

    # cache speed_factor ต่อประเภทถนน ลดการพยากรณ์ซ้ำ
    sf_cache = {}
    for _, _, _, d in G2.edges(keys=True, data=True):
        h = normalize_highway(d.get('highway', 'unclassified'))
        if h not in sf_cache:
            sf_cache[h] = predict_speed_factor(model, hour, h)

    name_keys = set(intervals_by_name.keys())
    to_remove = []

    for i, (u, v, k, data) in enumerate(G2.edges(keys=True, data=True), 1):
        # ชื่อถนนอาจเป็น list
        name = data.get('name', None)
        if isinstance(name, list):
            name = name[0]

        # เช็คเฉพาะถนนที่มีใน CSV เพื่อลด cost
        if name:
            key = str(name).strip().lower()
            if key in name_keys:
                if not is_open(name, dt_local, intervals_by_name):
                    to_remove.append((u, v, k))
                    # โชว์ความคืบหน้าบ้าง
                    if i % 20000 == 0:
                        elapsed = time.perf_counter() - start
                        print(f"  processed {i:,}/{total:,} edges | removed {len(to_remove):,} | {elapsed:.1f}s", flush=True)
                    continue

        base_tt = data.get('travel_time')
        if base_tt is None:
            length_m = data.get('length', 0.0)
            speed_kph = data.get('speed_kph', 30.0)
            base_tt = (length_m / 1000.0) / max(1e-3, speed_kph) * 3600.0

        f = sf_cache.get(normalize_highway(data.get('highway', 'unclassified')), 1.0)
        data['weight'] = float(base_tt / max(0.1, f))

        if i % 20000 == 0:
            elapsed = time.perf_counter() - start
            print(f"  processed {i:,}/{total:,} edges | removed {len(to_remove):,} | {elapsed:.1f}s", flush=True)

    for e in to_remove:
        try:
            G2.remove_edge(*e)
        except:
            pass

    elapsed = time.perf_counter() - start
    print(f"  removed closed edges: {len(to_remove):,} | total edges: {total:,} | took {elapsed:.1f}s", flush=True)
    return G2

def geocode_point(q):
    """
    แปลงสตริงเป็น (lat, lon)
    - ถ้าเป็น "lat,lon" จะ parse ตรงๆ
    - ถ้าเป็นชื่อสถานที่ ใช้ osmnx.geocode()
    """
    if isinstance(q, str) and ',' in q:
        lat, lon = [float(x.strip()) for x in q.split(',', 1)]
        return (lat, lon)
    lat, lon = ox.geocode(q)
    return (lat, lon)

def nearest_nodes_from_points(G, origin_q, dest_q):
    o_lat, o_lon = geocode_point(origin_q)
    d_lat, d_lon = geocode_point(dest_q)
    orig_node = ox.nearest_nodes(G, X=[o_lon], Y=[o_lat])[0]
    dest_node = ox.nearest_nodes(G, X=[d_lon], Y=[d_lat])[0]
    return orig_node, dest_node, (o_lat, o_lon), (d_lat, d_lon)

def collapse_to_simple_digraph(G, weight='weight'):
    """แปลง MultiDiGraph -> DiGraph โดยเก็บ edge น้ำหนักต่ำสุดต่อคู่ (u,v)"""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))  # คง attrs ของโหนดไว้
    for u, v, k, data in G.edges(keys=True, data=True):
        w = data.get(weight, data.get('travel_time', 0.0))
        if H.has_edge(u, v):
            if w < H[u][v].get('weight', float('inf')):
                # อัปเดตเป็น edge ที่ดีกว่า
                H[u][v].update({
                    'weight': w,
                    'length': data.get('length', 0.0),
                    'name': data.get('name'),
                    'highway': data.get('highway'),
                    'geometry': data.get('geometry'),
                    'travel_time': data.get('travel_time'),
                })
        else:
            H.add_edge(u, v, weight=w,
                       length=data.get('length', 0.0),
                       name=data.get('name'),
                       highway=data.get('highway'),
                       geometry=data.get('geometry'),
                       travel_time=data.get('travel_time'))
    return H



def k_best_paths(G, source, target, k=2, weight='weight'):
    """คืน k เส้นทางสั้นสุดบนกราฟ (ถ้าเป็น MultiDiGraph จะยุบเป็น DiGraph ก่อน)"""
    if G.is_multigraph():
        G = collapse_to_simple_digraph(G, weight=weight)
    gen = nx.shortest_simple_paths(G, source, target, weight=weight)
    paths = []
    try:
        for _ in range(k):
            paths.append(next(gen))
    except StopIteration:
        pass
    return paths

def path_summary(G, path):
    total_len = 0.0
    total_time = 0.0
    names = []
    for u, v in zip(path[:-1], path[1:]):
        # ถ้า MultiDiGraph ให้เลือก edge ที่ weight ต่ำสุดระหว่างคู่นี้
        best_d = None
        best_w = None
        for k, data in G.get_edge_data(u, v).items():
            w = data.get('weight', data.get('travel_time', 0.0))
            if best_w is None or w < best_w:
                best_w = w
                best_d = data
        total_time += best_w or 0.0
        total_len += (best_d or {}).get('length', 0.0)
        nm = (best_d or {}).get('name', None)
        if nm:
            if isinstance(nm, list): nm = nm[0]
            names.append(str(nm))
    return {
        'meters': float(total_len),
        'km': float(total_len) / 1000.0,
        'seconds': float(total_time),
        'minutes': float(total_time) / 60.0,
        'roads': names[:20]
    }

def plot_routes(G, routes, filepath='routes.png'):
    import matplotlib.pyplot as plt
    fig, ax = ox.plot_graph_routes(G, routes, route_linewidth=4, node_size=0, show=False, close=False)
    fig.savefig(filepath, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return filepath

def best_edge_data(G, u, v):
    """เลือก edge ระหว่าง (u,v) ที่มี weight ต่ำสุด (รองรับ MultiDiGraph)"""
    ed = G.get_edge_data(u, v)
    if not isinstance(ed, dict):
        return ed
    best_k, best_d, best_w = None, None, None
    for k, d in ed.items():
        w = d.get('weight', d.get('travel_time', 0.0))
        if best_w is None or w < best_w:
            best_k, best_d, best_w = k, d, w
    return best_d

def route_to_latlon_coords(G, route):
    """แปลงลิสต์ node ของ route -> ลิสต์พิกัด [(lat,lon), ...] ต่อเนื่อง"""
    coords = []
    for u, v in zip(route[:-1], route[1:]):
        d = best_edge_data(G, u, v)
        if d and ('geometry' in d and d['geometry'] is not None):
            # shapely LineString: coords เป็น (x,y) = (lon,lat)
            for x, y in list(d['geometry'].coords):
                if not coords or coords[-1] != (y, x):
                    coords.append((y, x))
        else:
            # ไม่มี geometry -> เส้นตรงระหว่างสองโหนด
            y1, x1 = G.nodes[u]['y'], G.nodes[u]['x']
            y2, x2 = G.nodes[v]['y'], G.nodes[v]['x']
            if not coords or coords[-1] != (y1, x1):
                coords.append((y1, x1))
            if coords[-1] != (y2, x2):
                coords.append((y2, x2))
    return coords



def save_folium_map(G, routes, origin_ll, dest_ll, filepath='routes.html'):
    import folium
    # เบสแมพ
    center = [(origin_ll[0] + dest_ll[0]) / 2.0, (origin_ll[1] + dest_ll[1]) / 2.0]
    m = folium.Map(location=center, zoom_start=13, tiles='cartodbpositron')

    # วาดโครงข่ายถนนแบบบางๆ (ถ้าใหญ่ไปจะช้า—ตัดทิ้งได้)
    try:
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        folium.GeoJson(
            edges_gdf[['geometry']].to_json(),
            name='Network',
            style_function=lambda x: {'color': '#B8BCC3', 'weight': 1, 'opacity': 0.6}
        ).add_to(m)
    except Exception:
        pass

    # วาดเส้นทาง
    colors = ['#E74C3C', '#2ECC71', '#3498DB', '#9B59B6']
    for i, r in enumerate(routes):
        coords = route_to_latlon_coords(G, r)  # << ใช้ของเราเอง ไม่พึ่ง ox.utils_graph
        folium.PolyLine(locations=coords, weight=6, opacity=0.95,
                        color=colors[i % len(colors)], tooltip=f"Route {i+1}").add_to(m)

    # หมุดต้น-ปลาย
    folium.Marker(origin_ll, icon=folium.Icon(color='blue', icon='play'), tooltip='Origin').add_to(m)
    folium.Marker(dest_ll,   icon=folium.Icon(color='green', icon='flag'), tooltip='Destination').add_to(m)

    # ซูมอัตโนมัติ
    m.fit_bounds([origin_ll, dest_ll])
    m.save(filepath)
    return filepath



# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Songkhla routing with time-windowed roads + ML speed factor")
    parser.add_argument('--origin', required=True, help='Origin: "lat,lon" หรือชื่อสถานที่ (ใช้ Nominatim geocode)')
    parser.add_argument('--destination', required=True, help='Destination: "lat,lon" หรือชื่อสถานที่')
    parser.add_argument('--depart', required=True, help='วันเวลาออกเดินทางในท้องถิ่น เช่น "2025-10-21 17:30"')
    parser.add_argument('--tz', default='Asia/Bangkok', help='โซนเวลา (ดีฟอลต์ Asia/Bangkok)')
    parser.add_argument('--schedule_csv', default='road_schedules.csv', help='CSV เวลาเปิด–ปิดถนน')
    parser.add_argument('--k', type=int, default=2, help='จำนวนเส้นทางที่ต้องการ (ดีฟอลต์ 2)')
    parser.add_argument('--no-plot', action='store_true', help='ไม่ต้องสร้างรูปภาพเส้นทาง')
    parser.add_argument('--place', default='Songkhla Province, Thailand', help='ขอบเขต OSM (ดีฟอลต์ทั้งจังหวัดสงขลา)')
    args = parser.parse_args()

    tz = pytz.timezone(args.tz)
    dt_local = tz.localize(datetime.strptime(args.depart, "%Y-%m-%d %H:%M"))

    print(f"[1/6] โหลดกราฟ OSM: {args.place}")
    G = load_graph(args.place)
    print(f"     nodes={len(G):,}  edges={G.number_of_edges():,}")

    print("[2/6] เทรนโมเดล ML (จำลอง)...")
    model = build_ml_model(G)

    print(f"[3/6] อ่านตารางเวลาเปิด–ปิด: {args.schedule_csv}")
    intervals_by_name = parse_schedule(args.schedule_csv)

    print(f"[4/6] สร้างกราฟแบบเปิดเท่านั้น ณ {dt_local.isoformat()} และคำนวณ weight แบบ dynamic")
    Gw = build_open_weighted_graph(G, dt_local, intervals_by_name, model)

    print("[5/6] หาโหนดต้นทาง/ปลายทางที่ใกล้ที่สุด...")
    orig_node, dest_node, orig_ll, dest_ll = nearest_nodes_from_points(Gw, args.origin, args.destination)

    print(f"[6/6] หา {args.k} เส้นทางที่ดีที่สุด (ตามเวลาที่คาดการณ์)")
    routes = k_best_paths(Gw, orig_node, dest_node, k=args.k, weight='weight')
    if not routes:
        print("❌ ไม่พบเส้นทางที่เป็นไปได้ในช่วงเวลานี้ (กราฟอาจขาดเพราะถนนปิด)")
        sys.exit(2)

    for i, r in enumerate(routes, 1):
        s = path_summary(Gw, r)
        print(f"\nRoute {i}: ≈ {s['km']:.2f} km, ≈ {s['minutes']:.1f} นาที")
        if s['roads']:
            print("  ตัวอย่างชื่อถนน:", " > ".join(s['roads']))
        else:
            print("  (ไม่มีชื่อถนนในบางช่วง)")

    if not args.no_plot:
        fp = plot_routes(Gw, routes, filepath='routes.png')
        print(f"\n🖼 บันทึกรูปเส้นทาง: {fp}")

    # เสริมไฟล์แผนที่โต้ตอบได้
    try:
        html = save_folium_map(Gw, routes, orig_ll, dest_ll, filepath='routes.html')
        print(f"🌐 แผนที่โต้ตอบได้: {html}")
    except Exception as e:
        print(f"(ข้าม folium: {e})")


if __name__ == "__main__":
    main()
