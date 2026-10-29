import asyncio
import queue
import threading

import rclpy
import websockets
from rclpy.node import Node
from std_msgs.msg import String

from worker_mission_control.logic.ws_protocol import parse_message


class MegadroneBridgeNode(Node):
	def __init__(self):
		super().__init__("megadrone_bridge_node")
		
		self.declare_parameter("megadrone_websocket_url", "ws://localhost:8765")
		self.declare_parameter("megadrone_reconnect_delay", 5.0)
		
		self.incoming = queue.Queue()

		self.incoming = queue.Queue()
		self.incoming_pub = self.create_publisher(String, "megadrone/incoming", 10)
		self.drain_timer = self.create_timer(0.01, self.drain_queue)
		
		self.loop = asyncio.new_event_loop()
		self.ws_thread = threading.Thread(target = self.run_loop, daemon = True)
		self.ws_thread.start()
	
	def run_loop(self):
		asyncio.set_event_loop(self.loop)
		self.loop.run_until_complete(self.listen())
	
	async def listen(self):
		url = self.get_parameter("megadrone_websocket_url").value
		delay = self.get_parameter("megadrone_reconnect_delay").value
		
		while rclpy.ok():
			try:
				async with websockets.connect(url) as ws:
					self.get_logger().info(f"connected to {url}")
					
					async for raw in ws:
						self.incoming.put(raw)
			except (OSError, websockets.ConnectionClosed) as e:
				self.get_logger().warn(f"websocket error: {e}, retrying in {delay}s")
				await asyncio.sleep(delay)
	
	def drain_queue(self):
		while not self.incoming.empty():
			parsed = parse_message(self.incoming.get_nowait())
			
			if parsed is None:
				continue
			
			msg = String()
			msg.data = parsed
			self.incoming_pub.publish(msg)


def main():
	rclpy.init()
	
	node = MegadroneBridgeNode()
	rclpy.spin(node)
	
	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main()