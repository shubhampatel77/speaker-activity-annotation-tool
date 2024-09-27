def load_goturn_model(init_fc=None):
    # model_path = os.path.join(goturn_path, 'goturn', 'models', 'pretrained', 'caffenet_weights.npy')
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Model file not found at {model_path}")
    model = GoturnNetwork(goturn_path, init_fc=init_fc).to(device)
    return model

def bbox_to_goturn(bbox):
    # if isinstance(bbox, list):
    return BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3])
    # else:
    #     return bbox  # If bbox is already a BoundingBox object

def goturn_to_bbox(bbox):
    return [bbox.x1, bbox.y1, bbox.x2, bbox.y2]


def prepare_input(image, bbox, target_size=(227, 227)):
    if isinstance(image, np.ndarray):
        # Convert OpenCV image (numpy array) to PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, str):
        # Load image from file path
        image = Image.open(image).convert('RGB')
    else:
        raise TypeError("image must be a numpy array or a file path")

    # Ensure bbox is a BoundingBox object
    if not isinstance(bbox, BoundingBox):
        bbox = BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3])

    # Compute the search region
    output_width = bbox.compute_output_width()
    output_height = bbox.compute_output_height()
    edge_spacing_x = bbox.edge_spacing_x()
    edge_spacing_y = bbox.edge_spacing_y()

    center_x = bbox.get_center_x()
    center_y = bbox.get_center_y()

    roi_left = max(0, center_x - output_width / 2 + edge_spacing_x)
    roi_top = max(0, center_y - output_height / 2 + edge_spacing_y)
    roi_right = min(image.width, center_x + output_width / 2 + edge_spacing_x)
    roi_bottom = min(image.height, center_y + output_height / 2 + edge_spacing_y)

    # Crop the image using the computed search region
    cropped_image = image.crop((roi_left, roi_top, roi_right, roi_bottom))

    # Resize the cropped image to the target size
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(cropped_image)
    return image_tensor.unsqueeze(0).to(device)

def expand_bbox(bbox, expansion_factor=1.3):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    new_x1 = max(0, center_x - new_width / 2)
    new_y1 = max(0, center_y - new_height / 2)
    new_x2 = min(1, center_x + new_width / 2)
    new_y2 = min(1, center_y + new_height / 2)
    return [new_x1, new_y1, new_x2, new_y2]

def compute_iou(box, boxes):
    # Convert inputs to numpy arrays if they're not already
    box = np.array(box)
    boxes = np.array(boxes)

    # Ensure boxes is 2D
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]

    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def compute_iou_single(box1, box2):
    # This function computes IoU between two single boxes
    return compute_iou(box1, np.array(box2).reshape(1, -1))[0]


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # Update the indices
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def detect_and_process_faces(frame, width, height, nms_threshold):
    try:
        detections = RetinaFace.detect_faces(frame)
    except Exception as e:
        logging.error(f"Error in face detection: {str(e)}")
        return []

    current_detections = [
        {
            'bbox': expand_bbox([coord / (width if i % 2 == 0 else height) for i, coord in enumerate(face['facial_area'])]),
            'score': face['score'],
            'landmarks': face['landmarks']
        }
        for face in detections.values()
    ]

    boxes = np.array([d['bbox'] for d in current_detections])
    scores = np.array([d['score'] for d in current_detections])
    keep = nms(boxes, scores, nms_threshold)
    return [current_detections[i] for i in keep]

def predict_track_positions(tracks, frame, goturn_network):
    predicted_boxes = {}
    for track_id, track in tracks.items():
        prev_bbox = bbox_to_goturn(track['bbox'])
        prev_frame = track['last_frame']
        
        prev_sample = prepare_input(prev_frame, prev_bbox).to(device)  # Move input to device
        curr_sample = prepare_input(frame, prev_bbox).to(device) 

        with torch.no_grad():
            pred_bbox = goturn_network(prev_sample, curr_sample)
        pred_bbox = pred_bbox.cpu().numpy().squeeze()
        predicted_boxes[track_id] = BoundingBox(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3])
    return predicted_boxes

def compute_iou_matrix(predicted_boxes, current_detections):
    iou_matrix = np.zeros((len(predicted_boxes), len(current_detections)))
    for i, (track_id, pred_box) in enumerate(predicted_boxes.items()):
        for j, det in enumerate(current_detections):
            iou_matrix[i, j] = compute_iou_single(goturn_to_bbox(pred_box), det['bbox'])
    return iou_matrix


def remove_old_tracks(tracks, frame_idx, max_frames_to_skip):
    tracks_to_remove = [track_id for track_id, track in tracks.items() 
                        if frame_idx - track['last_seen'] > max_frames_to_skip]
    
    for track_id in tracks_to_remove:
        del tracks[track_id]


def draw_and_save_frame_data(frame, tracks, width, height, fps, frame_idx):
    frame_data = []
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = [int(coord * (width if i % 2 == 0 else height)) for i, coord in enumerate(track['bbox'])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = str(track_id)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1 + 5
        label_y = y1 + label_size[1] + 5

        cv2.rectangle(frame, (label_x - 2, label_y - label_size[1] - 2), 
                      (label_x + label_size[0] + 2, label_y + 2), 
                      (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        frame_data.append({
            'track_id': track_id,
            'frame_timestamp': frame_idx / fps,
            'bbox': [float(coord) for coord in track['bbox']],
            'score': float(track['score']),
            'landmarks': {k: [float(v[0]), float(v[1])] for k, v in track['landmarks'].items()}
        })
    return frame_data

def match_tracks_detections(tracks, current_detections, iou_matrix, iou_threshold=0.3):
    track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
    
    matched_tracks = {}
    unmatched_detections = set(range(len(current_detections)))

    for track_idx, det_idx in zip(track_indices, detection_indices):
        if iou_matrix[track_idx, det_idx] >= iou_threshold:
            track_id = list(tracks.keys())[track_idx]
            matched_tracks[track_id] = current_detections[det_idx]
            unmatched_detections.remove(det_idx)
    
    return matched_tracks, unmatched_detections

def update_tracks(tracks, matched_tracks, frame_idx, frame):
    for track_id, detection in matched_tracks.items():
        tracks[track_id].update({
            'bbox': detection['bbox'],
            'score': detection['score'],
            'landmarks': detection['landmarks'],
            'last_seen': frame_idx,
            'last_frame': frame.copy()
        })

def create_new_tracks(tracks, unmatched_detections, current_detections, frame_idx, frame, next_track_id):
    for det_idx in unmatched_detections:
        detection = current_detections[det_idx]
        track_id = str(next_track_id)
        next_track_id += 1
        
        tracks[track_id] = {
            'bbox': detection['bbox'],
            'score': detection['score'],
            'landmarks': detection['landmarks'],
            'last_seen': frame_idx,
            'last_frame': frame.copy()
        }
    return next_track_id
        
def process_video(input_path, output_video_path, output_json_path):
    if os.path.exists(output_video_path) and os.path.exists(output_json_path):
        logging.info(f"Skipping {input_path} as output files already exist.")
        return

    cap = cv2.VideoCapture(input_path)
    goturn_network = load_goturn_model()
    goturn_network.eval()
    goturn_network = goturn_network.to(device)

    if not cap.isOpened():
        logging.error(f"Error opening video file: {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width == 0 or height == 0 or total_frames == 0:
        logging.error(f"Invalid video properties for {input_path}")
        return

    output_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)
    
    tracks = {}
    next_track_id = 0
    iou_threshold = 0.3
    nms_threshold = 0.5
    json_data = {}

    try:
        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_path)}") as pbar:
            for frame_idx in range(100,200):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                current_detections = detect_and_process_faces(frame, width, height, nms_threshold)
                
                if tracks:
                    predicted_boxes = predict_track_positions(tracks, frame, goturn_network)
                    iou_matrix = compute_iou_matrix(predicted_boxes, current_detections)
                    matched_tracks, unmatched_detections = match_tracks_detections(tracks, current_detections, iou_matrix, iou_threshold)
                    update_tracks(tracks, matched_tracks, frame_idx, frame)
                else:
                    unmatched_detections = set(range(len(current_detections)))
                
                next_track_id = create_new_tracks(tracks, unmatched_detections, current_detections, frame_idx, frame, next_track_id)
                
                frame_data = draw_and_save_frame_data(frame, tracks, width, height, fps, frame_idx)
                
                json_data[str(frame_idx)] = frame_data
                out.write(frame)
                pbar.update(1)

    except Exception as e:
        logging.error(f"Error processing video {input_path}: {str(e)}")
        raise
    finally:
        cap.release()
        out.release()

    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Processed {total_frames} frames")
    print(f"Annotated video saved to {output_video_path}")
    print(f"Face tracks data saved to {output_json_path}")
    
    
    # FIRST DETECTION CODE-------------------------------
    def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def expand_bbox(bbox, expansion_factor=1.3):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    new_x1 = max(0, center_x - new_width / 2)
    new_y1 = max(0, center_y - new_height / 2)
    new_x2 = min(1, center_x + new_width / 2)
    new_y2 = min(1, center_y + new_height / 2)
    return [new_x1, new_y1, new_x2, new_y2]

def process_video(input_path, output_video_path, output_json_path):
    if os.path.exists(output_video_path) and os.path.exists(output_json_path):
        logging.info(f"Skipping {input_path} as output files already exist.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width == 0 or height == 0 or total_frames == 0:
        logging.error(f"Invalid video properties for {input_path}")
        return

    output_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    tracks = {}
    next_track_id = 0
    iou_threshold = 0.6
    json_data = {}

    try:
        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_path)}") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    logging.warning(f"End of video or invalid frame at index {frame_idx} for {input_path}")
                    break

                try:
                    detections = RetinaFace.detect_faces(frame)
                except Exception as e:
                    logging.error(f"Error in face detection at frame {frame_idx} for {input_path}: {str(e)}")
                    continue

                frame_data = []
                current_tracks = {}

                for face_id, face in detections.items():
                    facial_area = face['facial_area']
                    landmarks = face['landmarks']
                    score = face['score']

                    x1, y1, x2, y2 = facial_area
                    current_bbox_unnormalized = [x1, y1, x2, y2]

                    # Normalize for storage and display
                    norm_x1, norm_y1 = x1 / width, y1 / height
                    norm_x2, norm_y2 = x2 / width, y2 / height
                    current_bbox_normalized = expand_bbox([norm_x1, norm_y1, norm_x2, norm_y2])
                    # current_bbox_unnormalized = expand_bbox(current_bbox_unnormalized)

                    best_match_id = None
                    best_iou = 0

                    for track_id, track in tracks.items():
                        iou = compute_iou(track['bbox'], current_bbox_normalized)
                        if iou > best_iou and iou >= iou_threshold:
                            best_match_id = track_id
                            best_iou = iou

                    if best_match_id is not None:


                        track_id = best_match_id
                        tracks[track_id]['bbox'] = current_bbox_normalized
                    else:
                        track_id = f"{next_track_id}"
                        next_track_id += 1
                        tracks[track_id] = {'bbox': current_bbox_normalized}

                    current_tracks[track_id] = tracks[track_id]

                    # Drawing and labeling (using normalized coordinates)
                    x1, y1, x2, y2 = [int(coord * (width if i % 2 == 0 else height)) for i, coord in enumerate(current_bbox_normalized)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = str(track_id)
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Calculate label position (always inside the box at top-left)
                    label_x = x1 + 5  # 5 pixels padding from the left edge of the box
                    label_y = y1 + label_size[1] + 5  # 5 pixels padding from the top edge of the box

                    # Draw a filled rectangle behind the text for better visibility
                    cv2.rectangle(frame, (label_x - 2, label_y - label_size[1] - 2), 
                                  (label_x + label_size[0] + 2, label_y + 2), 
                                  (0, 255, 0), cv2.FILLED)

                    # Put the label text
                    cv2.putText(frame, label, (label_x, label_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    frame_data.append({
                        'track_id': track_id,
                        'frame_timestamp': frame_idx / fps,
                        'bbox': [float(coord) for coord in current_bbox_normalized],
                        'score': float(score),
                        'landmarks': {k: [float(v[0]), float(v[1])] for k, v in landmarks.items()}
                    })

    except Exception as e:
        logging.error(f"Error processing video {input_path}: {str(e)}")
    finally:
        cap.release()
        out.release()

    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f"Processed {total_frames} frames")
    print(f"Annotated video saved to {output_video_path}")
    print(f"Face tracks data saved to {output_json_path}")