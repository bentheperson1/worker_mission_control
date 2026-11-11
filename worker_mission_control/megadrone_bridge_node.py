import asyncio
import json
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
		
		self.declare_parameter("megadrone_websocket_url", "ws://megadrone.local:8765")
		self.declare_parameter("megadrone_reconnect_delay", 5.0)
		
		self.incoming = queue.Queue()
		self.incoming_pub = self.create_publisher(String, "/megadrone/incoming", 10)
		self.drain_timer = self.create_timer(0.01, self.drain_queue)
		
		self.send_message_sub = self.create_subscription(String, "/megadrone/send", self.send_message, 10)
		
		self.loop = asyncio.new_event_loop()
		self.outgoing = asyncio.Queue()
		self.stop_event = asyncio.Event()
		
		self.ws_thread = threading.Thread(target=self.run_loop, daemon=True)
		self.ws_thread.start()
	
	def run_loop(self):
		asyncio.set_event_loop(self.loop)
		
		try:
			self.loop.run_until_complete(self.listen())
		except Exception as e:
			self.get_logger().error(f"listener thread died: {type(e).__name__}: {e}")
		finally:
			self.loop.close()
	
	async def listen(self):
		url = self.get_parameter("megadrone_websocket_url").value
		delay = self.get_parameter("megadrone_reconnect_delay").value
		
		while not self.stop_event.is_set():
			try:
				async with websockets.connect(url) as ws:
					self.get_logger().info(f"connected to {url}")
					
					tasks = [
						asyncio.create_task(self.receive_messages(ws)),
						asyncio.create_task(self.send_messages(ws)),
						asyncio.create_task(self.stop_event.wait()),
					]
					
					done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
					
					for task in pending:
						task.cancel()
					
					await asyncio.gather(*pending, return_exceptions=True)
					
					for task in done:
						task.result()
				
				if not self.stop_event.is_set():
					self.get_logger().warn(f"websocket closed, retrying in {delay}s")
			except Exception as e:
				self.get_logger().warn(f"websocket error: {type(e).__name__}: {e}, retrying in {delay}s")
			
			try:
				await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
			except asyncio.TimeoutError:
				pass
	
	async def receive_messages(self, ws):
		async for raw in ws:
			self.incoming.put(raw)
	
	async def send_messages(self, ws):
		while True:
			msg = await self.outgoing.get()
			await ws.send(msg)
	
	def drain_queue(self):
		while not self.incoming.empty():
			raw = self.incoming.get_nowait()
			parsed = parse_message(raw)
			
			if parsed is None:
				self.get_logger().warn(f"parse_message dropped: {raw}", throttle_duration_sec=2.0)
				continue
			
			if not isinstance(parsed, str):
				parsed = json.dumps(parsed)
			
			msg = String()
			msg.data = parsed
			self.incoming_pub.publish(msg)
	
	def send_message(self, msg):
		if self.loop.is_closed():
			return
		
		self.loop.call_soon_threadsafe(self.outgoing.put_nowait, msg.data)
	
	def destroy_node(self):
		if self.ws_thread.is_alive():
			try:
				self.loop.call_soon_threadsafe(self.stop_event.set)
			except RuntimeError:
				pass
			
			self.ws_thread.join(timeout=5.0)
		
		super().destroy_node()


def main():
	rclpy.init()
	
	node = MegadroneBridgeNode()
	
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == "__main__":
	main()