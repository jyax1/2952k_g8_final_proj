import re

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

def recolor_robot(xml_string: str, target_rgba: str = "0 0 0 1") -> str:
    """
    Given a MuJoCo XML string, update every visual <geom> element belonging to the robot
    so that its RGBA is set to target_rgba.
    
    We assume that robot visual geoms have names that start with "robot0_g" and contain "_vis".
    
    Returns the modified XML string.
    """
    # Parse the XML string into an ElementTree
    root = ET.fromstring(xml_string)
    
    # Iterate over all <geom> elements.
    for geom in root.iter("geom"):
        # We only want to override geoms that are for visual rendering.
        # We'll check if:
        #   - The geom has group="1" (convention for visual geoms)
        #   - Its name indicates it is a robot visual geom, e.g. name starts with "robot0_g" and includes "_vis"
        name = geom.get("name", "")
        group = geom.get("group", "")
        if group == "1" and re.search(r"^robot0_g.*_vis", name):
            # Override or insert the rgba attribute.
            geom.set("rgba", target_rgba)
    
    # Convert the tree back to a string.
    return ET.tostring(root, encoding="unicode")