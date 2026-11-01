import sys
from streamlit.testing.v1 import AppTest

def test_dashboard_officer_and_citizen_views():
    print("Testing dashboard loading in Municipal Officer View...")
    at = AppTest.from_file("dashboard/app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Exception in default run: {at.exception}"
    print("Officer View loaded successfully with 0 exceptions!")

    # Check that Walkthrough buttons exist and can be clicked
    print("Testing Walkthrough Tour buttons...")
    # Buttons: wt1, wt2, wt3, wt4
    stage_buttons = [b for b in at.button if "Stage" in b.label]
    assert len(stage_buttons) == 3, f"Expected 3 stage buttons, found {len(stage_buttons)}"
    
    # Click Stage 1
    stage_buttons[0].click().run()
    assert not at.exception, f"Exception after clicking Stage 1: {at.exception}"
    print("Stage 1 clicked cleanly!")

    # Click Stage 2
    stage_buttons = [b for b in at.button if "Stage" in b.label]
    stage_buttons[1].click().run()
    assert not at.exception, f"Exception after clicking Stage 2: {at.exception}"
    print("Stage 2 clicked cleanly!")

    # Click Stage 3
    stage_buttons = [b for b in at.button if "Stage" in b.label]
    stage_buttons[2].click().run()
    assert not at.exception, f"Exception after clicking Stage 3: {at.exception}"
    print("Stage 3 clicked cleanly!")

    # Check Perspective Toggle to Citizen View
    print("Testing toggle to Citizen / Commuter View...")
    role_radios = [r for r in at.radio if "Audience Perspective" in r.label]
    assert len(role_radios) == 1, "Audience Perspective radio not found"
    role_radios[0].set_value("🚶 Citizen / Commuter View").run()
    assert not at.exception, f"Exception in Citizen View: {at.exception}"
    print("Citizen View rendered successfully with 0 exceptions!")

    # Switch back to Municipal Officer View
    role_radios = [r for r in at.radio if "Audience Perspective" in r.label]
    role_radios[0].set_value("👔 Municipal Officer View").run()
    assert not at.exception, f"Exception switching back: {at.exception}"
    print("Switched back to Officer View cleanly!")

    print("ALL FEATURES TESTED AND VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_dashboard_officer_and_citizen_views()
