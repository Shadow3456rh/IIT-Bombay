"""
Layer 2 -- Drone Data Augmentation Pipeline
============================================
Module : src/output/layer2_json.py

PURPOSE
-------
Constructs the final Layer-2 JSON data package from existing Object3D definitions
and ONRelationResult evidence. Provides serialization logic and validation.

LIMITATIONS
-----------
- Only supports the "ON" geometric relationship prototype.
- No GNN integrations.
- Layer classification (static/dynamic) defaults to None.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from objects.object_3d import Object3D
from relationships.on_relation import ONRelationResult


@dataclass(frozen=True)
class Layer2ObjectRecord:
    object_id: int
    semantic_class_id: int
    class_name: str
    layer: Optional[str]
    gaussian_ids: List[int]
    gaussian_count: int
    center_3d: List[float]
    bbox_3d: Dict[str, List[float]]
    mean_confidence: Optional[float]
    relationships: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "semantic_class_id": self.semantic_class_id,
            "class_name": self.class_name,
            "layer": self.layer,
            "gaussian_ids": self.gaussian_ids,
            "gaussian_count": self.gaussian_count,
            "center_3d": self.center_3d,
            "bbox_3d": self.bbox_3d,
            "mean_confidence": self.mean_confidence,
            "relationships": self.relationships,
        }


@dataclass(frozen=True)
class Layer2Output:
    objects: List[Layer2ObjectRecord]
    layer_name: str = "Layer2"
    version: str = "0.1"
    coordinate_system: str = "Layer-1 world coordinates"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer_name,
            "version": self.version,
            "coordinate_system": self.coordinate_system,
            "objects": [obj.to_dict() for obj in self.objects],
        }


def _validate_xyz(coords: List[float], name: str) -> None:
    if len(coords) != 3:
        raise ValueError(f"{name} must have exactly 3 coordinates.")
    for c in coords:
        if not np.isfinite(c):
            raise ValueError(f"{name} contains non-finite values.")

def build_layer2_output(
    objects: List[Object3D],
    on_relations: List[ONRelationResult]
) -> Layer2Output:
    """
    Constructs the final Layer2Output payload.

    Parameters
    ----------
    objects : List[Object3D]
        The 3D objects recovered from Step 10.
    on_relations : List[ONRelationResult]
        The extracted relations from Step 11.

    Returns
    -------
    Layer2Output
    """
    
    object_ids = set()
    for obj in objects:
        if obj.object_id in object_ids:
            raise ValueError(f"Duplicate object ID {obj.object_id} detected.")
        object_ids.add(obj.object_id)
        
        # Verify 3D geometry
        _validate_xyz(obj.center_3d, f"Object {obj.object_id} center_3d")
        _validate_xyz(obj.bbox_min, f"Object {obj.object_id} bbox_min")
        _validate_xyz(obj.bbox_max, f"Object {obj.object_id} bbox_max")
        
        if any(obj.bbox_min[i] > obj.bbox_max[i] for i in range(3)):
            raise ValueError(f"Object {obj.object_id} has bbox_min > bbox_max")

    # Map relationships to subject IDs
    rel_map: Dict[int, List[Dict[str, Any]]] = {oid: [] for oid in object_ids}

    for rel in on_relations:
        if rel.subject_object_id not in object_ids:
            raise ValueError(f"Relation references unknown subject ID {rel.subject_object_id}")
        if rel.support_object_id not in object_ids:
            raise ValueError(f"Relation references unknown target ID {rel.support_object_id}")
            
        if rel.relation != "ON":
            raise ValueError(f"Unsupported relationship type: {rel.relation}")
            
        # Only attach the relation if it was actually detected as valid
        if rel.is_on:
            rel_dict = {
                "relation": rel.relation,
                "target_object_id": rel.support_object_id,
                "target_class_name": rel.support_class_name,
                "vertical_gap": round(rel.vertical_gap, 6),
                "overlap_ratio": round(rel.overlap_ratio, 6),
                "on_score": round(rel.on_score, 6),
            }
            rel_map[rel.subject_object_id].append(rel_dict)
            
    records = []
    
    # Sort for deterministic output
    for obj in sorted(objects, key=lambda x: x.object_id):
        # Package bounding box
        bbox = {
            "min": [float(c) for c in obj.bbox_min],
            "max": [float(c) for c in obj.bbox_max]
        }
        
        # Construct record
        record = Layer2ObjectRecord(
            object_id=obj.object_id,
            semantic_class_id=obj.semantic_class_id,
            class_name=obj.class_name,
            layer=None,  # Layer classification unavailable, missing data policy: None
            gaussian_ids=list(obj.gaussian_ids),
            gaussian_count=obj.gaussian_count,
            center_3d=[float(c) for c in obj.center_3d],
            bbox_3d=bbox,
            mean_confidence=float(obj.mean_confidence) if obj.mean_confidence is not None else None,
            relationships=rel_map[obj.object_id]
        )
        records.append(record)
        
    return Layer2Output(objects=records)


def build_extended_layer2_output(
    objects: List[Object3D],
    relationships: List[Dict[str, Any]],
    version: str = "0.2"
) -> Layer2Output:
    """
    Constructs an extended Layer2Output payload supporting multi-class relationship predictions
    (e.g., ON, UNDER, INSIDE, NEXT_TO, IN_FRONT_OF, BEHIND) with documented provenance and scores.
    Preserves exact object record structure for backward compatibility.
    """
    object_ids = set()
    for obj in objects:
        if obj.object_id in object_ids:
            raise ValueError(f"Duplicate object ID {obj.object_id} detected.")
        object_ids.add(obj.object_id)

        _validate_xyz(obj.center_3d, f"Object {obj.object_id} center_3d")
        _validate_xyz(obj.bbox_min, f"Object {obj.object_id} bbox_min")
        _validate_xyz(obj.bbox_max, f"Object {obj.object_id} bbox_max")

        if any(obj.bbox_min[i] > obj.bbox_max[i] for i in range(3)):
            raise ValueError(f"Object {obj.object_id} has bbox_min > bbox_max")

    rel_map: Dict[int, List[Dict[str, Any]]] = {oid: [] for oid in object_ids}

    for rel in relationships:
        subj_id = rel.get("subject_object_id")
        target_id = rel.get("target_object_id")
        if subj_id not in object_ids:
            raise ValueError(f"Relation references unknown subject ID {subj_id}")
        if target_id not in object_ids:
            raise ValueError(f"Relation references unknown target ID {target_id}")

        rel_entry: Dict[str, Any] = {
            "relation": rel.get("relation"),
            "target_object_id": target_id,
            "target_class_name": rel.get("target_class_name"),
            "method": rel.get("method", "gnn"),
            "score": round(float(rel["score"]), 6) if rel.get("score") is not None else None,
        }
        if "details" in rel and isinstance(rel["details"], dict):
            rel_entry["details"] = rel["details"]
        rel_map[subj_id].append(rel_entry)

    records = []
    for obj in sorted(objects, key=lambda x: x.object_id):
        bbox = {
            "min": [float(c) for c in obj.bbox_min],
            "max": [float(c) for c in obj.bbox_max],
        }
        record = Layer2ObjectRecord(
            object_id=obj.object_id,
            semantic_class_id=obj.semantic_class_id,
            class_name=obj.class_name,
            layer=None,
            gaussian_ids=list(obj.gaussian_ids),
            gaussian_count=obj.gaussian_count,
            center_3d=[float(c) for c in obj.center_3d],
            bbox_3d=bbox,
            mean_confidence=float(obj.mean_confidence) if obj.mean_confidence is not None else None,
            relationships=rel_map[obj.object_id],
        )
        records.append(record)

    return Layer2Output(objects=records, version=version)
