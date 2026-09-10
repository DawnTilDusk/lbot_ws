"""Conservative ring detector and camera-frame surface localization.

No robot control, ranking-based size labels or hole-center depth lookup.
Quality scores are heuristics, not calibrated probabilities.
"""
import cv2
import numpy as np


def rays(pixels, info):
    k = np.asarray(info['k'], dtype=float).reshape(3, 3)
    d = np.asarray(info.get('d', []), dtype=float)
    model = info.get('distortion_model', '')
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
    if model in ('plumb_bob', 'rational_polynomial', ''):
        xy = cv2.undistortPoints(pixels, k, d if d.size else None).reshape(-1, 2)
    elif model == 'equidistant' and d.size == 4:
        xy = cv2.fisheye.undistortPoints(pixels, k, d).reshape(-1, 2)
    else:
        raise ValueError('Unsupported camera distortion model: ' + model)
    return np.column_stack((xy, np.ones(len(xy))))


def fit_plane(points, tolerance):
    rng = np.random.default_rng(21)
    if len(points) > 700:
        points = points[rng.choice(len(points), 700, replace=False)]
    best = None
    for _ in range(70):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        offset = -np.dot(normal, a)
        inliers = np.abs(points @ normal + offset) < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 25:
        return None
    selected = points[best]
    center = np.mean(selected, axis=0)
    _, singular, vh = np.linalg.svd(selected-center, full_matrices=False)
    if singular[1] < 0.002:
        return None
    normal = vh[-1]
    if normal @ center > 0:
        normal = -normal
    offset = -normal @ center
    residual = np.abs(points @ normal + offset)
    inliers = residual < tolerance
    return normal, offset, float(np.mean(inliers)), float(np.sqrt(np.mean(residual[inliers]**2)))


def surface_position(depth, info, outer, hole, hole_center, config):
    result = {'position_valid': False, 'position_camera_m': None,
              'surface_normal_camera': None, 'across_flats_mm': None,
              'point_definition': 'hole_center_ray_intersection_with_nut_surface_plane',
              'invalid_reasons': []}
    mask = np.zeros(depth.shape, np.uint8)
    cv2.drawContours(mask, [outer], -1, 255, -1)
    cv2.drawContours(mask, [hole], -1, 0, -1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    yy, xx = np.nonzero(mask)
    if len(xx) < 35:
        result['invalid_reasons'].append('surface_ring_too_small')
        return result
    z = depth[yy, xx].astype(float) * config['depth_scale_m']
    valid = np.isfinite(z) & (z > 0.15) & (z < 4.0)
    result['depth_support_fraction'] = float(valid.mean())
    if valid.sum() < 35 or valid.mean() < config['min_depth_support']:
        result['invalid_reasons'].append('insufficient_valid_surface_depth')
        return result
    xx, yy, z = xx[valid], yy[valid], z[valid]
    points = rays(np.column_stack((xx, yy)), info) * z[:, None]
    plane = fit_plane(points, config['plane_tolerance_m'])
    if plane is None:
        result['invalid_reasons'].append('surface_plane_fit_failed')
        return result
    normal, offset, fraction, rms = plane
    result.update(plane_inlier_fraction=fraction, plane_rms_mm=rms*1000)
    inliers = np.abs(points @ normal + offset) < config['plane_tolerance_m']
    quadrants = set(((xx[inliers] > hole_center[0]).astype(int) +
                     2*(yy[inliers] > hole_center[1]).astype(int)).tolist())
    if fraction < 0.60 or len(quadrants) < 3:
        result['invalid_reasons'].append('surface_depth_inconsistent_or_one_sided')
        return result
    center_ray = rays([hole_center], info)[0]
    denom = normal @ center_ray
    if abs(denom) / np.linalg.norm(center_ray) < 0.35:
        result['invalid_reasons'].append('surface_view_too_oblique')
        return result
    point = center_ray * (-offset / denom)
    if not np.isfinite(point).all() or not 0.15 < point[2] < 4.0:
        result['invalid_reasons'].append('invalid_plane_intersection')
        return result
    # Project contour onto the fitted top plane, then measure minimum caliper width.
    edge_rays = rays(outer.reshape(-1, 2), info)
    edge_den = edge_rays @ normal
    if np.any(np.abs(edge_den) < 0.1):
        result['invalid_reasons'].append('unstable_contour_projection')
        return result
    cloud = edge_rays * (-offset / edge_den)[:, None]
    tangent = np.cross(normal, [1., 0., 0.])
    if np.linalg.norm(tangent) < 0.1:
        tangent = np.cross(normal, [0., 1., 0.])
    tangent /= np.linalg.norm(tangent)
    second = np.cross(normal, tangent)
    uv = np.column_stack(((cloud-point) @ tangent, (cloud-point) @ second))
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    widths = []
    for i in range(len(hull)):
        edge = hull[(i+1) % len(hull)] - hull[i]
        length = np.linalg.norm(edge)
        if length > 1e-6:
            axis = np.array([-edge[1], edge[0]]) / length
            projected = hull @ axis
            widths.append(float(np.ptp(projected)))
    width = min(widths)*1000 if widths else None
    if width is None or not 20 < width < 110:
        result['invalid_reasons'].append('implausible_nut_size')
        return result
    result.update(position_valid=True, position_camera_m=point.tolist(),
                  surface_normal_camera=normal.tolist(), across_flats_mm=width)
    return result


def classify(width, color, config):
    if width is None or color == 'unknown':
        return 'unknown', 'no_reliable_metric_size_or_color'
    matches = []
    for name, profile in config['profiles'].items():
        if config['task_mode'] == 'basic' and name == 'M45_black':
            continue
        reference = profile['across_flats_mm']
        if profile['color'] == color and reference and reference > 0:
            matches.append((abs(width-reference)/reference, name))
    if not matches:
        return 'unknown', 'measured_size_reference_missing'
    matches.sort()
    if matches[0][0] > config['size_relative_tolerance']:
        return 'unknown', 'size_outside_reference_tolerance'
    if len(matches) > 1 and matches[1][0]-matches[0][0] < 0.05:
        return 'unknown', 'ambiguous_size'
    return matches[0][1], 'measured_size_and_surface_color'


def projected_size_evidence(depth, info, outer, hole, config):
    """Estimate a size interval without assuming the silhouette is the top face.

    A regular hexagon's caliper width lies between AF and 2*AF/sqrt(3).
    The apparent hole major-axis direction reduces plane foreshortening.
    This is classification evidence only, never a substitute for a surface pose.
    """
    result = {'size_interval_mm': None, 'projected_span_mm': None,
              'size_evidence_valid': False, 'size_evidence_reason': 'no_depth_scale'}
    mask = np.zeros(depth.shape, np.uint8)
    cv2.drawContours(mask, [outer], -1, 255, -1)
    cv2.drawContours(mask, [hole], -1, 0, -1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    z = depth[mask > 0].astype(float) * config['depth_scale_m']
    valid = np.isfinite(z) & (z > .15) & (z < 4.)
    if len(z) < 25 or valid.sum() < 20 or valid.mean() < .20:
        return result
    z = z[valid]
    distance = float(np.median(z))
    relative_spread = float((np.percentile(z, 75)-np.percentile(z, 25))/distance)
    result.update(size_scale_depth_m=distance, size_depth_relative_iqr=relative_spread)
    if relative_spread > .15:
        result['size_evidence_reason'] = 'surface_depth_scale_inconsistent'
        return result
    ellipse = cv2.fitEllipse(hole)
    angle = np.deg2rad(ellipse[2] + (90 if ellipse[1][1] > ellipse[1][0] else 0))
    axis = np.array([np.cos(angle), np.sin(angle)])
    normalized = rays(outer.reshape(-1, 2), info)[:, :2]
    span = float(np.ptp(normalized @ axis) * distance * 1000)
    result.update(projected_span_mm=span, size_interval_mm=[span*np.sqrt(3)/2, span],
                  size_evidence_valid=True, size_evidence_reason='hexagon_projected_width_interval')
    return result


def classify_interval(evidence, appearance, config):
    if not evidence['size_evidence_valid']:
        return 'unknown', evidence['size_evidence_reason']
    if appearance == 'unknown':
        return 'unknown', 'surface_color_ambiguous'
    low, high = evidence['size_interval_mm']
    margin = .04
    candidates = []
    for name, profile in config['profiles'].items():
        width = profile.get('across_flats_mm')
        if width and profile['color'] == appearance and low*(1-margin) <= width <= high*(1+margin):
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0], 'metric_size_interval_and_surface_color'
    return 'unknown', 'ambiguous_size_interval' if candidates else 'no_matching_measured_size'


def contour_candidates(color, config):
    """Associate hole edges with silhouettes across thresholds, not only one tree."""
    h, w = color.shape[:2]
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(cv2.Canny(blur, 35, 100), cv2.MORPH_CLOSE, kernel)
    low = cv2.morphologyEx(cv2.Canny(blur, 15, 45), cv2.MORPH_CLOSE, kernel)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 7)
    masks = [edges, low, adaptive] + [cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)[1] for t in (55, 80, 110, 140, 170)]
    pools = [cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0] for mask in masks]
    holes = []
    for pool in pools[:3]:
        for contour in pool:
            area = cv2.contourArea(contour)
            if len(contour) < 5 or not 25 < area < h*w*.025:
                continue
            ellipse = cv2.fitEllipse(contour)
            a, b = ellipse[1]
            if min(a, b) < 5 or min(a, b)/max(a, b) < .28:
                continue
            error = abs(area/(np.pi*a*b/4)-1)
            if error > .20:
                continue
            center = np.array(ellipse[0])
            holes.append((area, contour, center, error))
    centers = np.array([hole[2] for hole in holes])
    if not len(centers):
        return [], edges
    x0,y0,x1,y1 = [v*s for v,s in zip(config.get('roi',[0,0,1,1]),[w,h,w,h])]
    candidates = []
    for source, pool in enumerate(pools):
        for outer in pool:
            area = cv2.contourArea(outer)
            if not 100 < area < h*w*.06:
                continue
            x,y,bw,bh = cv2.boundingRect(outer)
            if x<=x0+2 or y<=y0+2 or x+bw>=x1-2 or y+bh>=y1-2:
                continue
            _, axes, _ = cv2.minAreaRect(outer)
            short, long = sorted(axes)
            if short < 12 or short/long < .32:
                continue
            solidity = area/max(cv2.contourArea(cv2.convexHull(outer)), 1)
            if solidity < .83:
                continue
            # The specified task uses light paper/table support. Use local
            # appearance, not a fixed image rectangle, to reject dark machinery.
            left,top,right,bottom=max(0,x-8),max(0,y-8),min(w,x+bw+8),min(h,y+bh+8)
            local=np.zeros((bottom-top,right-left),np.uint8)
            shifted=outer-np.array([[[left,top]]],dtype=outer.dtype)
            cv2.drawContours(local,[shifted],-1,255,-1)
            surround=(cv2.dilate(local,np.ones((9,9),np.uint8))>0)&(local==0)
            background=gray[top:bottom,left:right][surround]
            if background.size<20 or np.median(background)<125:
                continue
            polygon = cv2.approxPolyDP(outer, .025*cv2.arcLength(outer,True), True)
            if not 5 <= len(polygon) <= 9:
                continue
            moment = cv2.moments(outer)
            center = np.array([moment['m10']/moment['m00'],moment['m01']/moment['m00']])
            possible = np.flatnonzero(np.linalg.norm(centers-center,axis=1)<short*.35)
            matches = []
            for index in possible:
                ha,hole,hc,error = holes[index]
                if not .12 < ha/area < .67:
                    continue
                if cv2.pointPolygonTest(outer,tuple(float(v) for v in hc),False)<0:
                    continue
                # Reject a partly external ellipse even if its center is inside.
                probe=hole.reshape(-1,2)[::max(1,len(hole)//12)]
                if sum(cv2.pointPolygonTest(outer,tuple(float(v) for v in pt),False)>=0 for pt in probe)<len(probe)*.9:
                    continue
                matches.append((1-error-np.linalg.norm(hc-center)/short+.1*ha/area,hole,hc))
            if not matches:
                continue
            _,hole,hc=max(matches,key=lambda item:item[0])
            score=float(.55*solidity+.25*(1-np.linalg.norm(hc-center)/short)+.20*(5<=len(polygon)<=7))
            candidates.append((score,area,outer,hole,hc,(x,y,bw,bh),source))
    # Cluster alternate contours, then choose a representative near the median
    # area rather than letting one high-contrast shadow dominate the outline.
    groups=[]
    for candidate in sorted(candidates,key=lambda c:c[0],reverse=True):
        group=next((group for group in groups if np.linalg.norm(candidate[4]-group[0][4])<.4*min(candidate[5][2:]+group[0][5][2:])),None)
        if group is None:groups.append([candidate])
        else:group.append(candidate)
    selected=[]
    for group in groups:
        representative=max(group,key=lambda c:c[1])
        selected.append(representative[:6])
    return selected,edges


def detect(color, depth, info, config):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    selected, edges = contour_candidates(color, config)
    results = []
    for score, area, outer, hole, hc, box in selected:
        ring = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(ring, [outer], -1, 255, -1)
        cv2.drawContours(ring, [hole], -1, 0, -1)
        ring = cv2.erode(ring, np.ones((3,3),np.uint8))
        values = gray[ring>0]
        median = float(np.median(values)) if values.size else 0
        appearance = 'black' if median < 80 else ('silver' if median > 115 else 'unknown')
        evidence = projected_size_evidence(depth, info, outer, hole, config)
        # Reject hardware holes whose reliable metric scale is far below nut size.
        # Missing depth still leaves an explicitly unknown 2-D detection.
        known_widths = [p['across_flats_mm'] for p in config['profiles'].values() if p.get('across_flats_mm')]
        if evidence['size_evidence_valid'] and known_widths:
            low, high = evidence['size_interval_mm']
            if high < min(known_widths)*.75 or low > max(known_widths)*1.25:
                continue
        localization = surface_position(depth, info, outer, hole, hc, config)
        label, reason = classify_interval(evidence, appearance, config)
        allowed = ['M45_silver', 'M33_black', 'M27_black'] if config['task_mode'] == 'basic' else ['M45_silver', 'M45_black', 'M33_black', 'M27_black']
        result = {'class': label, 'class_valid': label != 'unknown', 'class_reason': reason,
                  'appearance': appearance, 'surface_gray_median': median,
                  'detection_quality': score, 'quality_is_probability': False,
                  'center_pixel': hc.tolist(), 'bbox_xywh': list(box),
                  'outer_contour': outer.reshape(-1,2).tolist(),
                  'hole_contour': hole.reshape(-1,2).tolist(),
                  'in_task': label in allowed, **localization, **evidence}
        results.append(result)
    priority = {'M45_silver': 0, 'M45_black': 1, 'M33_black': 2, 'M27_black': 3, 'unknown': 99}
    results.sort(key=lambda obj: (priority[obj['class']], -obj['detection_quality']))
    return results, edges


def annotate(image, detections, roi):
    out = image.copy()
    h,w = out.shape[:2]
    box = [int(v*s) for v,s in zip(roi,[w,h,w,h])]
    if not np.allclose(roi, [0.0, 0.0, 1.0, 1.0]):
        cv2.rectangle(out, tuple(box[:2]), tuple(box[2:]), (200,150,30), 2)
    for i,obj in enumerate(detections):
        color = (40,220,60) if obj['position_valid'] and obj['class_valid'] else (0,180,255)
        cv2.polylines(out,[np.array(obj['outer_contour'],np.int32)],True,color,2)
        cv2.polylines(out,[np.array(obj['hole_contour'],np.int32)],True,(255,160,0),1)
        x,y = [int(v) for v in obj['center_pixel']]
        cv2.drawMarker(out,(x,y),(0,0,255),cv2.MARKER_CROSS,12,2)
        width = obj['across_flats_mm']
        label = f"{i}: {obj['class']} / {obj['appearance']}"
        cv2.putText(out,label,(max(0,x-60),max(18,y-18)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
        if width:
            cv2.putText(out,f"AF={width:.1f}mm",(x,y+18),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    return out
