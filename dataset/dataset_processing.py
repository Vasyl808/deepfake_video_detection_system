import os
import cv2
import json
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Tuple, Union, Dict


def convert_path(path: str) -> str:
    for i in range(10):
        path = path.replace(f'{i:02}', str(i))
    return path


def process_file_paths(df: pd.DataFrame, base_path) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        file_path = row['file_path']
        full_path = os.path.join(base_path, file_path)
        result.append((full_path, row['label']))
    return result


def get_video_info(video_path: str) -> Union[Tuple[int, float], Tuple[None, None]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не вдалося відкрити файл: {video_path}")
        return None, None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count, fps


def df_to_list(df: pd.DataFrame) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        result.append((row['file_path'], row['label']))
    return result


def create_train_val_test_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_data = df[~df['video_path'].str.contains('dfdc_train_part_8|dfdc_train_part_7')]
    val_data = df[df['video_path'].str.contains('dfdc_train_part_8')]
    test_data = df[df['video_path'].str.contains('dfdc_train_part_7')]
    return train_data, val_data, test_data


def save_dataframe_to_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, index=False)


def create_json_from_csv(input_csv: str, output_json: str) -> None:
    df = pd.read_csv(input_csv)
    result: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        file_path = row['video_path']
        label = row['label']
        result[file_path] = {'label': label}
    with open(output_json, 'w') as json_file:
        json.dump(result, json_file, indent=4)


def count_labels_from_json(json_file: str) -> Dict[str, int]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    label_counts = Counter()
    for item in data.values():
        label_counts[item['label']] += 1
    return dict(label_counts)


def get_data_path(base_path: str) -> Tuple[str, str]:
    metadata_csv = base_path + 'faces_metadata.csv'
    df = pd.read_csv(metadata_csv)

    file_list = process_file_paths(df, base_path)

    video_stats = []
    for video_path, label in file_list:
        frame_count, fps = get_video_info(video_path)
        if frame_count is not None and fps is not None:
            video_stats.append({
                'video_path': video_path,
                'label': label,
                'frames': frame_count,
                'fps': fps
            })
    video_df = pd.DataFrame(video_stats)

    train_data, val_data, test_data = create_train_val_test_splits(video_df)

    print("Train set size:", len(train_data))
    print("Val set size:", len(val_data))
    print("Test set size:", len(test_data))

    save_dataframe_to_csv(train_data, "train_dataset.csv")
    save_dataframe_to_csv(val_data, "val_dataset.csv")
    save_dataframe_to_csv(test_data, "test_dataset.csv")

    create_json_from_csv('train_dataset.csv', 'output_balenced_train.json')
    create_json_from_csv('val_dataset.csv', 'output_balenced_val.json')
    create_json_from_csv('test_dataset.csv', 'output_balenced_test.json')

    counts = count_labels_from_json('output_balenced_train.json')
    print("Label counts for train dataset:")
    for label, count in counts.items():
        print(f"Label: {label}, Count: {count}")
    
    return 'output_balenced_train.json', 'output_balenced_val.json', 'output_balenced_test.json'
