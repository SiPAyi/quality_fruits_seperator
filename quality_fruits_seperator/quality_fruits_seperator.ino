#include <MotorDriver.h>
MotorDriver m;

int ledPin = 13;  // LED pin

void setup() {
  pinMode(ledPin, OUTPUT);  // Set LED pin as output
  Serial.begin(9600);       // Start serial communication at 9600 baud rate
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readString();  // Read incoming string

    // Check if the input string is "good"
    if (input == "good\n") {
      m.motor(4,FORWARD,255);  
      blinkGood();  // Blink pattern for "good"
    } 
    // Check if the input string is "bad"
    else if (input == "bad\n") {
      m.motor(4,BACKWARD,255);  
      blinkBad();   // Blink pattern for "bad"
    }
  }
}

// Blink sequence for "good" - fast blink
void blinkGood() {
  for (int i = 0; i < 15; i++) {  // Blink 3 times
    digitalWrite(ledPin, HIGH);  // Turn on LED
    delay(200);                  // Wait for 200 milliseconds
    digitalWrite(ledPin, LOW);   // Turn off LED
    delay(200);                  // Wait for 200 milliseconds
  }
}

// Blink sequence for "bad" - slow blink
void blinkBad() {
  for (int i = 0; i < 3; i++) {  // Blink 3 times
    digitalWrite(ledPin, HIGH);  // Turn on LED
    delay(1000);                 // Wait for 1 second
    digitalWrite(ledPin, LOW);   // Turn off LED
    delay(1000);                 // Wait for 1 second
  }
}
