import pygame
import time

def joystick_process(cmd_queue):
    """
    Reads joystick inputs continuously and sends the latest velocity
    and button commands to a multiprocessing queue.

    This process is intended to run in parallel with the AI controller
    and arbitration logic, providing real-time human control commands.
    """

    # Initialize pygame and joystick subsystem
    pygame.init()
    pygame.joystick.init()

    # Check if at least one joystick is connected
    if pygame.joystick.get_count() == 0:
        print("[ERROR] No joystick detected!")
        return

    # Use the first detected joystick
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print("Joystick connected:", joy.get_name())

    # Main loop: continuously read joystick state
    while True:
        # Update internal pygame event state
        pygame.event.pump()

        # ==========================
        # Read joystick axes
        # ==========================
        # Axis mapping (typical PS-style controller):
        # Axis 1 → forward/backward (vx)
        # Axis 0 → left/right (vy)
        # Axis 3 → yaw rotation
        vx  = -joy.get_axis(1) * 1.2
        vy  = -joy.get_axis(0) * 1.2
        yaw = -joy.get_axis(3) * 1.5

        # ==========================
        # Deadzone filtering
        # ==========================
        # Small joystick noise is ignored to avoid unintended motion
        if abs(vx)  < 0.12: vx  = 0
        if abs(vy)  < 0.12: vy  = 0
        if abs(yaw) < 0.12: yaw = 0

        # ==========================
        # Read button states
        # ==========================
        # Button 0 → A button
        # Button 1 → B button
        A = joy.get_button(0)
        B = joy.get_button(1)

        # ==========================
        # Queue management
        # ==========================
        # Remove all old commands so that only the
        # most recent human input is used
        while not cmd_queue.empty():
            try:
                cmd_queue.get_nowait()
            except:
                break

        # Send the latest command set to the arbitration layer
        cmd_queue.put({
            "vx": vx,
            "vy": vy,
            "yaw": yaw,
            "A": A,
            "B": B
        })

        # Run at ~50 Hz (20 ms loop)
        time.sleep(0.02)
