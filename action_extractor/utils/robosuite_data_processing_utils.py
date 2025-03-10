import re
import copy

def recolor_gripper(xml_string: str) -> str:
    """
    Given a MuJoCo XML string, replace the RGBA for:
      - gripper0_hand_visual => green (0 1 0 1)
      - gripper0_finger1_visual => cyan (0 1 1 1)
      - gripper0_finger2_visual => magenta (1 0 1 1)
    Returns a new XML string with the updated RGBA.
    """

    # Regex patterns with 3 capturing groups:
    #   (1) The part up to and including rgba="
    #   (2) The old color contents (we'll overwrite)
    #   (3) The closing quote "
    pattern_hand = r'(geom\s+name="gripper0_hand_visual".*?rgba=")([^"]*)(")'
    pattern_left_finger = r'(geom\s+name="gripper0_finger1_visual".*?rgba=")([^"]*)(")'
    pattern_right_finger = r'(geom\s+name="gripper0_finger2_visual".*?rgba=")([^"]*)(")'

    # Use \g<1> and \g<3> so we don't accidentally invoke \10
    replacement_hand = r'\g<1>0 1 0 1\g<3>'
    replacement_left_finger = r'\g<1>0 1 1 1\g<3>'
    replacement_right_finger = r'\g<1>1 0 1 1\g<3>'

    xml_string = re.sub(pattern_hand, replacement_hand, xml_string, flags=re.DOTALL)
    xml_string = re.sub(pattern_left_finger, replacement_left_finger, xml_string, flags=re.DOTALL)
    xml_string = re.sub(pattern_right_finger, replacement_right_finger, xml_string, flags=re.DOTALL)

    return xml_string

def replace_all_lights(xml_string: str) -> str:
    """
    1) Remove every self-closing <light .../> definition in the MuJoCo XML.
    2) Insert the new desired <light .../> lines right after <worldbody>.

    The inserted lines are:
      <light pos="1.0 0 1.5" ... />
      <light pos="1.0 0 0.8" ... />
      <light pos="-0.24 0. 0.80" ... />
      <light pos="0.0 1.0 1.5" ... />
      <light pos="0.0 0.0 0.8" ... />
      <light pos="0.0 -1.0 1.5" ... />
    """
    # 1) Remove all lines that match <light .../> (i.e. self-closing tags)
    #    We assume all lights are self-closing, e.g. <light .../>
    pattern_remove_lights = r'<light\b.*?/>'
    xml_string = re.sub(pattern_remove_lights, '', xml_string, flags=re.DOTALL)

    # 2) Define the new lights block
    #    Each is appended with "\n" so they appear on separate lines
    new_lights_block = (
        '<light pos="1.0 0 1.5" dir="-0.2 0.0 -1" diffuse="0.4 0.4 0.4" specular="0.4 0.4 0.4" directional="true" castshadow="false"/>\n'
        '<light pos="1.0 0 0.8" dir="-0.2 0.0 -0.6" diffuse="0.4 0.4 0.4" specular="0.4 0.4 0.4" directional="true" castshadow="false"/>\n'
        '<light pos="-0.24 0. 0.80" dir="1.0 0.0 0.0" diffuse="0.4 0.4 0.4" specular="0.4 0.4 0.4" directional="true" castshadow="false"/>\n'
        '<light pos="0.0 1.0 1.5" dir="0.0 -0.2 -1" diffuse="0.4 0.4 0.4" specular="0.4 0.4 0.4" directional="true" castshadow="false"/>\n'
        '<light pos="0.0 0.0 0.8" dir="0.0 0.0 1" diffuse="0.4 0.4 0.4" specular="0.4 0.4 0.4" directional="true" castshadow="false"/>\n'
        '<light pos="0.0 -1.0 1.5" dir="0.0 0.2 -1" diffuse="0.4 0.4 0.4" specular="0.4 0.4 0.4" directional="true" castshadow="false"/>\n'
    )

    # 3) Insert the new block immediately after <worldbody> (the first occurrence).
    #    This approach is simpler than matching a specific line for old lights.
    pattern_worldbody = r'(<worldbody>)'
    xml_string = re.sub(
        pattern_worldbody,
        r'\1\n' + new_lights_block,  # \1 is the captured <worldbody>
        xml_string,
        count=1  # only replace the first <worldbody> we find
    )

    return xml_string


def find_index_after_pattern(text, pattern, after_pattern):
    # Find the index of the first occurrence of after_pattern
    start_index = text.find(after_pattern)
    if start_index == -1:
        return -1
    
    # Search for pattern after the start_index
    index_after_pattern = text.find(pattern, start_index)
    if index_after_pattern == -1:
        return -1
    
    # Return the index after the pattern
    return index_after_pattern + len(pattern)


def insert_camera_info(xml_string: str) -> str:
    pattern = '/>\n'
    after_pattern = 'camera name="sideview"'

    insert_index = find_index_after_pattern(xml_string['model'], pattern, after_pattern) + 1

    new_cameras_xml =  '''<camera mode="fixed" name="sideview2" pos="0 -1.5 1.4879572214102434" quat="0.7933533 0.6087614 0 0" />\n    
                    <camera mode="fixed" name="backview" pos="-1.5 0 1.45" quat="-0.56 -0.43 0.43 0.56" />\n
                    <camera mode="fixed" name="sideagentview" pos="0 0.5 1.35" quat="0.0 0.0 0.383 0.923"/>\n
                    <camera mode="fixed" name="fronttableview" pos="0.8 0 1.2" quat="0.5608419  0.43064642 0.43064642 0.5608419"/>\n
                    <camera mode="fixed" name="sidetableview" pos="0 0.8 1" quat="0.01071808 0.00552625 0.69142354 0.72234905"/>\n
                    <camera mode="fixed" name="squared0view" pos="0.6 0.6 1" quat="0.28633323 0.26970193 0.63667727 0.6632619"/>\n
                    <camera mode="fixed" name="squared0viewfar" pos="0.9 0.9 1.0" quat="0.28633323 0.26970193 0.63667727 0.6632619"/>\n
                    <camera mode="fixed" name="squared0view2" pos="0.6 -0.6 1" quat="0.6714651  0.6409069  0.25949073 0.2665288"/>\n
                    <camera mode="fixed" name="squared0view2far" pos="0.9 -0.9 1" quat="0.6714651  0.6409069  0.25949073 0.2665288"/>\n
                    <camera mode="fixed" name="squared0view3" pos="-0.6 0.6 1" quat="-0.2665288  -0.25949073  0.6409069 0.6714651"/>\n
                    <camera mode="fixed" name="squared0view3far" pos="-0.9 0.9 1" quat="-0.2665288  -0.25949073  0.6409069 0.6714651"/>\n
                    <camera mode="fixed" name="squared0view4" pos="-0.6 -0.6 1" quat="0.6632619 0.63667727 -0.26970193 -0.28633323"/>\n
                    <camera mode="fixed" name="squared0view4far" pos="-0.9 -0.9 1" quat="0.6632619 0.63667727 -0.26970193 -0.28633323"/>\n
                    '''

    xml_string['model'] = xml_string['model'][:insert_index] + new_cameras_xml + xml_string['model'][insert_index:]
    
    return xml_string

import xml.etree.ElementTree as ET

def recolor_robot(xml_string: str, 
                  target_rgba: str = "0 0 0 1", 
                  target_specular: str = "0", 
                  target_shininess: str = "0") -> str:
    """
    Given a MuJoCo XML string, update every visual <geom> element belonging to the robot
    so that its inline RGBA attribute is set to target_rgba. In addition, update all
    <material> elements that define specular and shininess so that they are set to
    target_specular and target_shininess respectively.
    
    We assume that robot visual geoms have names that contain "_vis" and belong to group "1".
    
    Returns the modified XML string.
    """
    # Parse the XML string into an ElementTree.
    root = ET.fromstring(xml_string)
    
    # Update visual <geom> elements.
    # Only modify geoms with group="1" and a name that includes "_vis".
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        group = geom.get("group", "")
        if group == "1" and re.search(r"^robot0_g.*_vis", name):
            # Override or insert the rgba attribute.
            geom.set("rgba", target_rgba)
    
    # Update <material> elements to set specular and shininess.
    for material in root.iter("material"):
        mat_name = material.get("name", "")
        if mat_name.startswith("robot0_"):
            material.set("specular", target_specular)
            material.set("shininess", target_shininess)
    
    # Convert the modified tree back to a string.
    return ET.tostring(root, encoding="unicode")

def convert_robot_in_state(source_state, target_env, robot_prefix="robot0"):
    """
    Convert a state from one robot to another by replacing all robot-related 
    elements in the source XML with those from the target environment, and 
    also copy over the matte material used for the robot.

    Args:
        source_state (dict): Original state dict with keys "model" (XML string)
                             and "states" (flattened array).
        target_env: Environment whose robot configuration (and assets) you want to use.
        robot_prefix (str): Prefix for robot-related bodies (default "robot0").

    Returns:
        dict: Modified state with the updated model XML and updated state vector.
    """
    # 1) Get XML strings from source state and target environment.
    source_xml = source_state["model"]
    target_xml = target_env.env.sim.model.get_xml()

    # 2) Parse XML trees.
    source_tree = ET.fromstring(source_xml)
    target_tree = ET.fromstring(target_xml)

    # -------------------------------------------------------------------------
    # 3) Replace robot bodies in the <worldbody>, but do so "in place" so they
    #    appear in the same location (index) they were originally, rather than
    #    always at the end.
    # -------------------------------------------------------------------------
    source_worldbody = source_tree.find("worldbody")
    target_worldbody = target_tree.find("worldbody")

    # First, gather a list of all child bodies in source_worldbody
    source_bodies = list(source_worldbody)

    # Identify which indices correspond to robot bodies
    robot_indices = []
    for i, body in enumerate(source_bodies):
        name = body.get("name", "")
        if name.startswith(robot_prefix):
            robot_indices.append(i)

    # Remove these robot bodies from the list, going backwards so indices don't shift
    for i in reversed(robot_indices):
        del source_bodies[i]

    # Determine where in the list we want to re-insert the new robot bodies
    if robot_indices:
        # e.g. insert at the earliest occurrence
        insert_index = min(robot_indices)
    else:
        # if no robot bodies found, just append at the end
        insert_index = len(source_bodies)

    # Collect the corresponding robot bodies from the target
    target_robot_bodies = []
    for body in list(target_worldbody):
        name = body.get("name", "")
        if name.startswith(robot_prefix):
            # copy them so we don't mutate the original
            target_robot_bodies.append(copy.deepcopy(body))

    # Insert them back into the source bodies list at the chosen index
    for i, new_body in enumerate(target_robot_bodies):
        source_bodies.insert(insert_index + i, new_body)

    # Finally, overwrite the source_worldbody’s children with our updated list
    source_worldbody[:] = source_bodies

    # -------------------------------------------------------------------------
    # 4) Replace robot-related assets in <asset>, same as you already do.
    # -------------------------------------------------------------------------
    source_asset = source_tree.find("asset")
    target_asset = target_tree.find("asset")

    # Remove any asset (mesh or material) whose name starts with the robot prefix
    for elem in list(source_asset):
        if elem.tag in ["mesh", "material"]:
            if elem.get("name", "").startswith(robot_prefix):
                source_asset.remove(elem)

    # Append all robot-related asset elements from the target
    for elem in list(target_asset):
        if elem.tag in ["mesh", "material"]:
            if elem.get("name", "").startswith(robot_prefix):
                source_asset.append(copy.deepcopy(elem))
    
    # -------------------------------------------------------------------------
    # 5) Replace the entire <actuator> section with the target's actuator
    # -------------------------------------------------------------------------
    source_actuator = source_tree.find("actuator")
    target_actuator = target_tree.find("actuator")

    if source_actuator is not None:
        parent = source_tree  # <actuator> is direct child of <mujoco>
        parent.remove(source_actuator)

    if target_actuator is not None:
        # We try to insert it before <sensor> if present
        sensor = source_tree.find("sensor")
        actuator_copy = copy.deepcopy(target_actuator)
        if sensor is not None:
            parent = list(source_tree)
            sensor_index = parent.index(sensor)
            source_tree.insert(sensor_index, actuator_copy)
        else:
            source_tree.append(actuator_copy)

    # -------------------------------------------------------------------------
    # 6) Convert the modified tree back to a string, and build the new state dict
    # -------------------------------------------------------------------------
    new_xml = ET.tostring(source_tree, encoding="unicode")
    new_state = {
        "model": new_xml,
        # If you literally want the EXACT state from source_state, keep:
        "states": source_state['states']
        # Or if you want to adopt the target environment’s state vector:
        # "states": target_env.env.sim.get_state().flatten()
    }
    return new_state

import difflib
def compare_xml_strings(source_xml, new_xml):
    """
    Compare two XML strings and print their differences to the console.
    """
    # Convert each XML string into a list of lines
    source_lines = source_xml.splitlines(keepends=True)
    new_lines = new_xml.splitlines(keepends=True)

    # Use difflib.unified_diff to produce a unified diff
    diff = difflib.unified_diff(
        source_lines,
        new_lines,
        fromfile='source_xml',
        tofile='new_xml'
    )

    # Print the diff line by line
    for line in diff:
        print(line, end='')