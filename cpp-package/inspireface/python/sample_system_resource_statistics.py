
import inspireface as isf
import click
    
@click.command()
@click.argument("resource_path")
def case_show_system_resource_statistics(resource_path):
    """
    This case is used to test the system resource statistics.
    """
    ret = isf.launch(resource_path)
    assert ret, "Launch failure. Please ensure the resource path is correct."
    print("-" * 100)
    print("Initialization state")
    print("-" * 100)
    isf.show_system_resource_statistics()
    print("-" * 100)
    print("Create 10 sessions")
    print("-" * 100)
    print("")
    num_created_sessions = 10
    sessions = []
    for i in range(num_created_sessions):
        session = isf.InspireFaceSession(isf.HF_ENABLE_FACE_RECOGNITION, isf.HF_DETECT_MODE_ALWAYS_DETECT)
        sessions.append(session)
    isf.show_system_resource_statistics()
    print("-" * 100)
    print("Release 10 sessions")
    print("-" * 100)
    print()
    for session in sessions:
        session.release()
    isf.show_system_resource_statistics()

if __name__ == "__main__":
    case_show_system_resource_statistics()
