import asyncio
import websockets
import speech_recognition as sr

async def send_message():
    # Connect to the WebSocket server
    async with websockets.connect("ws://localhost:5000") as websocket:
        print("Connected to WebSocket server")

        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        while True:
            print("Listening...")
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
                audio = recognizer.listen(source)

            try:
                # Convert speech to text
                text = recognizer.recognize_google(audio)
                print(f"Recognized: {text}")

                # Send the text to the server
                await websocket.send(text)
                print(f"Sent: {text}")

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"STT error: {e}")

# Run the client
asyncio.get_event_loop().run_until_complete(send_message())