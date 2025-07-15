import os
import argparse
import random
import sys
import json
import cv2
import mediapipe as mp
import yt_dlp

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("Error: 'google-api-python-client' is not installed. Please run: pip install google-api-python-client")
    sys.exit(1)

try:
    from isodate import parse_duration
except ImportError:
    print("Error: 'isodate' is not installed. Please run: pip install isodate")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- OpenAI import and check ---
try:
    import openai
except ImportError:
    print("Error: 'openai' is not installed. Please run: pip install openai")
    sys.exit(1)
# --- End OpenAI import and check ---

# --- Add matplotlib import and check ---
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: 'matplotlib' is not installed. Please run: pip install matplotlib")
    sys.exit(1)
# --- End matplotlib import and check ---

# --- Verification Logic Configuration ---
VERIFY_TARGET_DURATION_SECONDS = 30
VERIFY_VISIBILITY_THRESHOLD = 0.7
VERIFY_MIN_VISIBLE_LANDMARKS_PERCENTAGE = 0.8 # 80% of essential landmarks must be visible
MOVEMENT_THRESHOLD = 0.3 # Adjust this value based on testing (e.g., average displacement of landmarks)
# Tolerance for bad frames within a 1-second segment (15%)
BAD_FRAME_TOLERANCE_RATIO = 0.15

mp_pose = mp.solutions.pose

# ANSI escape codes for colors
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_END = '\033[0m'

# --- Helper & Verification Functions ---

def parse_iso8601_duration(duration_string: str) -> int:
    return int(parse_duration(duration_string).total_seconds())

def download_video(url, temp_filename="temp_video.mp4"):
    """Download a video and return its path and duration."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]/best[height<=480]',
        'outtmpl': temp_filename,
        'quiet': True,
        'overwrite': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Step 1: Extract info without downloading
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            
            # Step 2: Perform the download
            ydl.download([url])

        return temp_filename, duration
    except Exception as e:
        print(f"-> yt-dlp error: {e}", file=sys.stderr)
        return None, 0

def is_full_body_visible(landmarks, threshold, min_visible_percentage):
    if not landmarks: return False
    essential_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    ]
    
    visible_count = 0
    for landmark in essential_landmarks:
        if landmarks.landmark[landmark].visibility >= threshold:
            visible_count += 1
            
    return (visible_count / len(essential_landmarks)) >= min_visible_percentage

def merge_segments(segments):
    if not segments:
        return []

    # Assumption: segments already respect camera‑cut boundaries; they are not merged across cuts.
    # Sort segments by their start time
    segments.sort(key=lambda x: x['start'])

    merged = []
    current_segment = segments[0]

    for i in range(1, len(segments)):
        next_segment = segments[i]
        # Check for overlap or contiguity (within 1 second for robustness)
        if next_segment['start'] <= current_segment['end'] + 1: # Allow 1 second gap for contiguity
            current_segment['end'] = max(current_segment['end'], next_segment['end'])
            # Merge displacement data if needed, this part needs careful consideration
            # For now, we'll just extend the segment.
            # If displacement data needs to be continuous, it would require more complex logic
            # to interpolate or concatenate based on the original video frames.
            # For this task, we'll assume the primary goal is merging time ranges.
        else:
            merged.append(current_segment)
            current_segment = next_segment
    
    merged.append(current_segment)
    return merged

# -------- Camera‑cut & movement helpers --------
def _frame_histogram(frame):
    """Return normalized grayscale histogram (256 bins)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist

def is_camera_dynamic(video_path, sample_interval=2, threshold=0.25):
    """
    Roughly classify camera as 'dynamic' or 'static' by comparing global
    appearance every `sample_interval` seconds.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(sample_interval * fps)
    prev_hist = None
    diffs = []
    for f in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break
        hist = _frame_histogram(frame)
        if prev_hist is not None:
            diff = 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diffs.append(diff)
        prev_hist = hist
    cap.release()
    if not diffs:
        return False  # default to static if unsure
    return (sum(diffs) / len(diffs)) > threshold  # True → dynamic

def is_camera_cut(prev_frame, curr_frame, cut_threshold=0.65):
    """
    Detect shot boundary by histogram correlation; returns True if cut detected.
    """
    if prev_frame is None or curr_frame is None:
        return False
    corr = cv2.compareHist(_frame_histogram(prev_frame),
                           _frame_histogram(curr_frame),
                           cv2.HISTCMP_CORREL)
    return corr < cut_threshold
# -------- End helpers --------

def verify_video_content(video_path, debug_mode=False, min_segment_duration=30):
    if debug_mode:
        print(f"DEBUG: verify_video_content called with debug_mode={debug_mode}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    if fps == 0: return None

    # Determine camera movement class once for the whole video
    camera_movement = 'dynamic' if is_camera_dynamic(video_path) else 'static'

    bad_frame_tolerance_per_second = int(fps * BAD_FRAME_TOLERANCE_RATIO)

    # List to store all successful segments found in this video
    successful_segments = []
    # --- New: List to store average displacement per 2-second chunk ---
    displacement_per_chunk = []
    # --- End New ---

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Stage 1: Coarse Scan - Check 1-second segments every 10 seconds
        # Store the 'G' or 'R' status for each *sampled* second.
        sampled_second_results = {} # {second_index: 'G'/'R'}

        print("--- Coarse Scan (1s segment every 10s) ---")
        for current_second_start in range(0, int(video_duration), 10):
            start_frame_of_segment = int(current_second_start * fps)
            end_frame_of_segment = int(min((current_second_start + 1) * fps, total_frames))

            bad_frames_in_segment = 0
            total_frames_in_segment = 0

            for frame_idx in range(start_frame_of_segment, end_frame_of_segment):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: break

                if not is_full_body_visible(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks, VERIFY_VISIBILITY_THRESHOLD, VERIFY_MIN_VISIBLE_LANDMARKS_PERCENTAGE):
                    bad_frames_in_segment += 1
                total_frames_in_segment += 1
            
            if total_frames_in_segment > 0 and bad_frames_in_segment <= bad_frame_tolerance_per_second:
                sampled_second_results[current_second_start] = 'G'
            else:
                sampled_second_results[current_second_start] = 'R'
            
            # Simple visualization for coarse scan
            # print(f"  Second {current_second_start} status: {sampled_second_results[current_second_start]}", end='\\r', flush=True)
        print("\n--- Coarse Scan Complete ---")

        # New: Visualize coarse scan results
        coarse_scan_bar = "".join([f"{C_GREEN}█{C_END}" if sampled_second_results[s] == 'G' else f"{C_RED}█{C_END}" for s in sorted(sampled_second_results.keys())])
        print(f"Coarse Scan Visualization: [{coarse_scan_bar}]")

        # Stage 2: Fine‑grained verification
        # Instead of a sliding fixed‑length window, walk through the region in
        # non‑overlapping 2‑second chunks, mark each chunk as good or bad, and
        # merge consecutive good chunks that are at least `min_segment_duration`
        # long.

        good_regions = []
        current_start = -1
        sorted_seconds = sorted(sampled_second_results.keys())

        for sec in sorted_seconds:
            if sampled_second_results[sec] == 'G':
                if current_start == -1:
                    current_start = sec
            else:
                if current_start != -1:
                    good_regions.append((current_start, sec))
                    current_start = -1
        if current_start != -1:
            good_regions.append((current_start, int(video_duration)))

        if debug_mode:
            print(f"Identified good regions from coarse scan: {good_regions}")

        for region_start, region_end in good_regions:
            region_len = region_end - region_start
            # Skip regions shorter than min_segment_duration right away
            if region_len < min_segment_duration:
                if debug_mode:
                    print(f"--- Skipping region {region_start}s to {region_end}s "
                          f"({region_len}s) < min_duration ({min_segment_duration}s) ---")
                continue

            print(f"--- Fine‑grained verification within good region: {region_start}s to {region_end}s ---")

            chunk_duration = 2                        # seconds per fine‑grained chunk
            current_chunk_start = region_start
            current_segment_start = None              # start time of an accumulating good segment
            current_segment_displacement = []         # displacement data for that segment

            while current_chunk_start + chunk_duration <= region_end:
                # ----- evaluate one 2‑second chunk -----
                start_frame_of_chunk = int(current_chunk_start * fps)
                end_frame_of_chunk   = int(min((current_chunk_start + chunk_duration) * fps, total_frames))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_of_chunk)

                bad_frames_in_chunk = 0
                total_frames_in_chunk = 0
                chunk_displacements = []
                prev_landmarks = None

                for _ in range(start_frame_of_chunk, end_frame_of_chunk):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    current_landmarks = results.pose_landmarks

                    # visibility check
                    is_visible = is_full_body_visible(
                        current_landmarks,
                        VERIFY_VISIBILITY_THRESHOLD,
                        VERIFY_MIN_VISIBLE_LANDMARKS_PERCENTAGE
                    )
                    if not is_visible:
                        bad_frames_in_chunk += 1

                    # displacement calc
                    if prev_landmarks and current_landmarks:
                        disp_sum = 0.0
                        for i in range(len(mp_pose.PoseLandmark)):
                            lm1 = prev_landmarks.landmark[i]
                            lm2 = current_landmarks.landmark[i]
                            disp_sum += ((lm2.x - lm1.x) ** 2 +
                                         (lm2.y - lm1.y) ** 2 +
                                         (lm2.z - lm1.z) ** 2) ** 0.5
                        chunk_displacements.append(disp_sum / len(mp_pose.PoseLandmark))

                    prev_landmarks = current_landmarks
                    total_frames_in_chunk += 1

                # decide if chunk passes
                bad_tolerance = int(fps * chunk_duration * BAD_FRAME_TOLERANCE_RATIO)
                chunk_passes = (total_frames_in_chunk > 0 and bad_frames_in_chunk <= bad_tolerance)

                # detect shot change between chunks
                cut_detected = False
                if 'prev_chunk_last_frame' in locals():
                    cut_detected = is_camera_cut(prev_chunk_last_frame, frame if ret else None)
                # store last frame of this chunk for next iteration
                prev_chunk_last_frame = frame if ret else None

                # ----- accumulate / flush segments -----
                if chunk_passes:
                    if current_segment_start is None:
                        current_segment_start = current_chunk_start
                    # store summary displacement for the chunk
                    current_segment_displacement.append(sum(chunk_displacements))
                else:
                    if current_segment_start is not None:
                        segment_end = current_chunk_start
                        segment_len = segment_end - current_segment_start
                        if segment_len >= min_segment_duration:
                            successful_segments.append({
                                "status": "SUCCESS",
                                "start":  current_segment_start,
                                "end":    segment_end,
                                "displacement_data": list(current_segment_displacement)
                            })
                            print(f"-> SUCCESS: Found segment from {current_segment_start}s to {segment_end}s.")
                        # reset accumulator
                        current_segment_start = None
                        current_segment_displacement = []

                # additionally break segment if camera cut occurred
                if cut_detected and current_segment_start is not None:
                    segment_end = current_chunk_start  # cut position
                    segment_len = segment_end - current_segment_start
                    if segment_len >= min_segment_duration:
                        successful_segments.append({
                            "status": "SUCCESS",
                            "start":  current_segment_start,
                            "end":    segment_end,
                            "displacement_data": list(current_segment_displacement)
                        })
                        if debug_mode:
                            print(f"-> CUT: Segment ended at camera cut {segment_end}s.")
                    current_segment_start = None
                    current_segment_displacement = []

                # advance to next chunk
                current_chunk_start += chunk_duration

            # handle trailing good segment that reaches region_end
            if current_segment_start is not None:
                segment_end = region_end
                segment_len = segment_end - current_segment_start
                if segment_len >= min_segment_duration:
                    successful_segments.append({
                        "status": "SUCCESS",
                        "start":  current_segment_start,
                        "end":    segment_end,
                        "displacement_data": list(current_segment_displacement)
                    })
                    print(f"-> SUCCESS: Found segment from {current_segment_start}s to {segment_end}s.")
            
    cap.release()
    
    # Merge overlapping segments
    merged_segments = merge_segments(successful_segments)
    return merged_segments, camera_movement

# --- New: Function to plot displacement graph ---
def plot_displacement_graph(displacement_data, video_duration, output_filename="displacement_graph.png"):
    if not displacement_data:
        print("No displacement data to plot.")
        return

    # Generate x-axis values (time in seconds, assuming 2-second chunks)
    x_values = [i * 2 for i in range(len(displacement_data))]
    
    # --- New: Calculate cumulative sum for the graph ---
    cumulative_displacement = [sum(displacement_data[:i+1]) for i in range(len(displacement_data))]

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, cumulative_displacement, marker='o', linestyle='-', markersize=4)
    plt.title('Cumulative Pose Displacement Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Displacement')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Displacement graph saved to {output_filename}")
    plt.close() # Close the plot to free memory
# --- End New ---

# --- Main Application Logic (largely unchanged) ---

def main():
    # ... (The main function remains the same as the previous version)
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    youtube = build('youtube', 'v3', developerKey=api_key)

    def load_used_queries_history():
        """Load history of all queries that have ever been used."""
        history_file = "used_queries_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def save_used_queries_history(used_queries):
        """Save history of all queries that have ever been used."""
        history_file = "used_queries_history.json"
        with open(history_file, "w") as f:
            json.dump(used_queries, f, ensure_ascii=False, indent=2)

    def add_to_used_queries_history(query):
        """Add a query to the history of used queries."""
        used_queries = load_used_queries_history()
        if query not in used_queries:
            used_queries.append(query)
            save_used_queries_history(used_queries)

    def generate_new_query_via_openai(used_queries=None):
        """Generate a new full‑body motion search query via OpenAI. Works with both openai<1.0 and >=1.0."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY is not set.")
            return None

        # Prepare duplicate‑avoidance list for the prompt
        used_queries = used_queries or []
        used_txt = "\n".join([f"- {q}" for q in used_queries]) if used_queries else ""
        avoid_block = f"\n\nDo NOT repeat any of these queries:\n{used_txt}" if used_txt else ""

        prompt = ("Generate one short YouTube search query (5‑10 words) likely to find videos "
                  "showing the entire human body in motion. Focus on diverse activities including: "
                  "daily activities (cooking, cleaning, walking, exercising), sports (dancing, yoga, "
                  "running, swimming), work activities (typing, lifting, reaching), and recreational "
                  "activities (playing, gardening, shopping). Return only the query text." + avoid_block)

        try:
            # ---- Try the new openai>=1.0 style first ----
            if hasattr(openai, "OpenAI"):                # openai>=1.0 provides a Client-like attribute
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.8,
                )
                return resp.choices[0].message.content.strip()

            # ---- Fall back to legacy <1.0 interface ----
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI query‑generation error: {e}", file=sys.stderr)
            return None
    # --- End helpers ---

    parser = argparse.ArgumentParser(description="Search, verify, and collect YouTube videos.")
    parser.add_argument('-q', '--query', nargs='+', help='Search query.')
    parser.add_argument('--verified-results', type=int, default=5, help='Number of *verified* videos to collect.')
    parser.add_argument('--min-duration', type=int, default=VERIFY_TARGET_DURATION_SECONDS, help=f'Minimum video duration to search for. Defaults to {VERIFY_TARGET_DURATION_SECONDS}s.')
    parser.add_argument('--output-file', default='verified_videos.json', help='Output JSON file name.')
    parser.add_argument('-n', '--max-fetch', type=int, default=25, help='Max videos to fetch from API per page.')
    parser.add_argument('--url', help='Direct URL of a YouTube video to verify.')
    parser.add_argument('--debug', action='store_true', help='Enable debug output.')
    parser.add_argument('--num-queries', type=int, default=10,
                        help='How many *successful* search queries to generate (default: 10).')
    parser.add_argument('--videos-per-query', type=int, default=2,
                        help='Target number of videos to collect per successful query (default: 2).')
    parser.add_argument('--tolerance', type=int, default=5,
                        help='Consecutive failures allowed for the active query before moving on (default: 5).')

    args = parser.parse_args()

    verified_videos = []

    if args.url:
        print(f"Verifying single video from URL: {args.url}")
        video_id = args.url.split('v=')[-1].split('&')[0] # Simple extraction, might need more robust parsing
        
        # For a single URL, we need to fetch its details to get duration
        try:
            video_details_request = youtube.videos().list(part="contentDetails,snippet", id=video_id)
            video_details_response = video_details_request.execute()
            if not video_details_response.get('items'):
                print(f"Error: Could not fetch details for video ID {video_id}. Skipping.")
                sys.exit(1)
            item = video_details_response['items'][0]
            duration_seconds = parse_iso8601_duration(item['contentDetails']['duration'])
            title = item['snippet']['title']
            url = args.url

            if duration_seconds < args.min_duration:
                print(f"Video duration ({duration_seconds}s) is less than minimum required duration ({args.min_duration}s). Skipping.")
                sys.exit(0)

            print(f"\n--- Verifying: {title} ({url}) ---")
            download_result = download_video(url)
            print(f"DEBUG: download_video returned: {download_result}")

            if not isinstance(download_result, tuple) or len(download_result) != 2:
                print("-> ERROR: download_video returned unexpected value. Skipping.", file=sys.stderr)
                sys.exit(1)
            
            temp_file, real_duration = download_result

            if not temp_file:
                print("-> Download failed.")
                sys.exit(1)
            
            # verify_video_content now returns a list of successful segments and camera_movement
            successful_segments, camera_movement = verify_video_content(temp_file, args.debug, args.min_duration)
            os.remove(temp_file)

            if successful_segments:
                # Use **all** segments that already satisfy min_duration
                qualified_segments = [
                    {
                        'start': seg['start'],
                        'end': seg['end'],
                        'displacement_data': []  # placeholder; fill if needed
                    }
                    for seg in successful_segments
                ]
                print(f"-> SUCCESS: Using {len(qualified_segments)} segment(s) ≥ {args.min_duration}s for this video.")

                video_data = {
                    'title': title,
                    'url': url,
                    'video_duration_seconds': duration_seconds,
                    'search_query': " ".join(args.query) if args.query else None,
                    'camera_movement': camera_movement,
                    'verified_segments': qualified_segments
                }
                verified_videos.append(video_data)

                # Generate (empty) displacement graphs for every segment for consistency
                for seg in qualified_segments:
                    graph_filename = f"displacement_graph_{video_id}_{seg['start']}s_{seg['end']}s.png"
                    plot_displacement_graph([], real_duration, output_filename=graph_filename)

                with open(args.output_file, 'w') as f:
                    json.dump(verified_videos, f, indent=4)

                print(f"*** Collected {len(verified_videos)} / {args.verified_results} verified videos. ***")
            else:
                print("-> FAILURE: No valid segment found in this video.")
                with open("log.txt", "a") as log_file:
                    log_file.write(f"Rejected: {title} ({url}) - Reason: No valid segment found.\n")
            
            print(f"\nProcess finished. Total verified videos collected: {len(verified_videos)}.")
            print(f"Results saved to {args.output_file}")
            sys.exit(0) # Exit after processing single URL

        except HttpError as e:
            print(f"An HTTP error occurred: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)

    # --- New OpenAI‑driven query loop ---
    total_queries_needed   = args.num_queries
    videos_per_query_target = args.videos_per_query
    tolerance_limit         = args.tolerance

    successful_query_count = 0
    used_queries_history   = load_used_queries_history()
    verified_videos        = []

    while successful_query_count < total_queries_needed:
        # Generate a fresh query with OpenAI
        search_query = generate_new_query_via_openai(used_queries_history)
        if not search_query:
            print("OpenAI failed to generate a query. Exiting.")
            break
        print(f"\n=== Generated query {successful_query_count+1}/{total_queries_needed}: '{search_query}' ===")
        add_to_used_queries_history(search_query)
        used_queries_history.append(search_query)

        videos_this_query = 0
        consecutive_failures = 0
        next_page_token = None

        while videos_this_query < videos_per_query_target and consecutive_failures < tolerance_limit:
            # ----- call YouTube search API -----
            try:
                search_request = youtube.search().list(
                    q=search_query, part='snippet', type='video',
                    videoLicense='creativeCommon', maxResults=args.max_fetch,
                    pageToken=next_page_token
                )
                search_response = search_request.execute()
                video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]

                if not video_ids:
                    print("No more results for this query.")
                    break

                video_details_request = youtube.videos().list(
                    part="contentDetails,snippet", id=",".join(video_ids))
                video_details_response = video_details_request.execute()

                any_success_in_page = False

                for item in video_details_response.get('items', []):
                    duration_seconds = parse_iso8601_duration(item['contentDetails']['duration'])
                    if duration_seconds < args.min_duration:
                        continue

                    video_id = item['id']
                    title    = item['snippet']['title']
                    url      = f"https://www.youtube.com/watch?v={video_id}"
                    print(f"\n--- Verifying: {title} ({url}) ---")

                    temp_file, real_duration = download_video(url)
                    if not temp_file:
                        continue

                    successful_segments, camera_movement = verify_video_content(
                        temp_file, args.debug, args.min_duration)
                    os.remove(temp_file)

                    if successful_segments:
                        qualified_segments = [{
                            'start': seg['start'],
                            'end':   seg['end'],
                            'displacement_data': []
                        } for seg in successful_segments]

                        video_data = {
                            'title': title,
                            'url': url,
                            'video_duration_seconds': duration_seconds,
                            'search_query': search_query,
                            'camera_movement': camera_movement,
                            'verified_segments': qualified_segments
                        }
                        verified_videos.append(video_data)
                        videos_this_query   += 1
                        consecutive_failures = 0
                        any_success_in_page  = True

                        with open(args.output_file, 'w') as f:
                            json.dump(verified_videos, f, indent=4)

                        print(f"✔ collected {videos_this_query}/{videos_per_query_target} "
                              f"videos for current query "
                              f"({len(verified_videos)} total).")
                        if videos_this_query >= videos_per_query_target:
                            break  # done for this query
                    else:
                        consecutive_failures += 1

                if any_success_in_page is False:
                    consecutive_failures += 1

                if consecutive_failures >= tolerance_limit:
                    print(f"Tolerance limit reached ({tolerance_limit}) for query '{search_query}'.")
                    break

                next_page_token = search_response.get('nextPageToken')
                if not next_page_token:
                    break
            except HttpError as e:
                print(f"HTTP error: {e}")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

        if videos_this_query > 0:
            successful_query_count += 1
            print(f"✅ Finished query '{search_query}' with {videos_this_query} videos.")
        else:
            print(f"❌ Query '{search_query}' yielded no videos.")

    # --- End new loop ---

    print(f"\nProcess finished."
          f" Successful queries: {successful_query_count}/{total_queries_needed}. "
          f"Total verified videos collected: {len(verified_videos)}.")
    print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
