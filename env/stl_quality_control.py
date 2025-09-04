from stl import mesh
import utils

def is_binary_stl(filename):
    """
    Check if an STL file is in binary format
    """
    try:
        # Try to read the file as a binary STL
        m = mesh.Mesh.from_file(filename)
        return True
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    transfered_stl_file = utils.convert_ascii_to_binary("G:\irregularBPP\dataset\objaversestl\Articulated_Shark_Plastic3D.stl", "G:\irregularBPP\dataset\objaversestl\Articulated_Shark_Plastic3D2.stl")
    print(is_binary_stl("G:\irregularBPP\dataset\objaversestl\Articulated_Shark_Plastic3D2.stl"))