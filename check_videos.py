
import argparse
import json
import subprocess
import os

def download_and_cut_clip(video_url, start_time, end_time, output_filename):
    """
    指定されたURLの動画をダウンロードし、指定された時間でカットして保存する。
    """
    print(f"Downloading and cutting clip from {video_url} from {start_time} to {end_time}...")

    # yt-dlpで動画をダウンロード
    # -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best はmp4形式で最高品質の動画と音声をダウンロード
    # --output temp_video.mp4 は一時ファイル名
    download_command = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--output", "temp_video.mp4",
        video_url
    ]
    try:
        subprocess.run(download_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        return

    # ffmpegで動画をカット
    # -ss は開始時間、-to は終了時間
    # -i は入力ファイル
    # -c:v copy -c:a copy は再エンコードせずにコピー
    # -avoid_negative_ts make_zero は負のタイムスタンプを0にする
    cut_command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", "temp_video.mp4",
        "-c:v", "copy",
        "-c:a", "copy",
        "-avoid_negative_ts", "make_zero",
        output_filename
    ]
    try:
        subprocess.run(cut_command, check=True)
        print(f"Clip saved as {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error cutting video: {e}")
    finally:
        # 一時ファイルを削除
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

def process_video(video_info, video_index, output_dir):
    """
    指定された動画情報を処理してクリップを作成する。
    """
    video_url = video_info["url"]
    video_title = video_info["title"].replace("/", "_").replace("\\", "_") # ファイル名に使用できない文字を置換

    print(f"Processing video {video_index}: {video_title} ({video_url})")

    for i, segment in enumerate(video_info["verified_segments"]):
        start_time = segment["start"]
        end_time = segment["end"]
        output_filename = os.path.join(output_dir, f"{video_title}_clip_{i+1}_{start_time}-{end_time}.mp4")
        download_and_cut_clip(video_url, start_time, end_time, output_filename)

def main():
    parser = argparse.ArgumentParser(description="Download and cut video clips from verified_videos.json.")
    parser.add_argument("--index", type=int, nargs='+', 
                        help="Index or indices of the video(s) in verified_videos.json to process. If not specified, all videos will be processed.")
    parser.add_argument("--output_dir", type=str, default="output_clips",
                        help="Directory to save the output clips.")

    args = parser.parse_args()

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    try:
        with open("verified_videos.json", "r") as f:
            verified_videos = json.load(f)
    except FileNotFoundError:
        print("Error: verified_videos.json not found.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode verified_videos.json. Check if it's valid JSON.")
        return

    # 処理するインデックスのリストを決定
    if args.index is None:
        # インデックスが指定されていない場合はすべての動画を処理
        indices_to_process = list(range(len(verified_videos)))
        print(f"No index specified. Processing all {len(verified_videos)} videos.")
    else:
        # 指定されたインデックスを処理
        indices_to_process = args.index
        # インデックスの範囲チェック
        for idx in indices_to_process:
            if not (0 <= idx < len(verified_videos)):
                print(f"Error: Index {idx} is out of bounds. There are {len(verified_videos)} videos.")
                return
        print(f"Processing {len(indices_to_process)} specified video(s).")

    # 指定されたインデックスの動画を処理
    for idx in indices_to_process:
        video_info = verified_videos[idx]
        process_video(video_info, idx, args.output_dir)

if __name__ == "__main__":
    main()
