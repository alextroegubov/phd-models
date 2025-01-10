from utils import ParametersSet
import itertools
import logging
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Callable, List
import random
from enum import Enum

from typing import Optional

test_params = ParametersSet(2, [15, 3], [1, 1], [1, 5], 1, 50, 1, 1, 50)


class EventType(Enum):
    REQUEST_ARRIVAL = 1
    REQUEST_SERVICE = 2
    STOP_SIMULATION = 3


@dataclass(order=True)
class Event:
    time: float
    action: Callable[[], None] | None = field(compare=False)  # event action
    event_type: EventType = field(compare=False)
    event_id: int = field(compare=False)  # Optional identifier for debugging


class Simulator:

    CANCELED_EVENT_TIME = -100

    def __init__(self):
        self.current_time = 0.0
        self.event_id_counter = 0
        self.event_queue = PriorityQueue()
        self.event_lookup: dict[int, Event] = {}

        self.processed_events = 0

    def _get_next_id(self):
        new_id = self.event_id_counter
        self.event_id_counter += 1
        return new_id

    def schedule_event(self, time: float, action: Callable[[], None], event_type: EventType) -> int:
        """Schedules an event at a specific simulation time."""
        event_id = self._get_next_id()
        event = Event(time, action, event_type, event_id)

        self.event_lookup[event_id] = event
        self.event_queue.put(event)
        return event_id

    def cancel_event(self, event_id: int):
        event = self.event_lookup.get(event_id)
        if event is not None:
            # delete from lookup table
            del self.event_lookup[event_id]
            # make field invalid for queue
            event.time = self.CANCELED_EVENT_TIME
            event.action = None  # lambda *args, **kwargs: None
        else:
            print("Cancel non-existing event")

    def run(self):
        """Starts the simulation loop."""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            if event.time != self.CANCELED_EVENT_TIME:
                self.current_time = event.time
                event.action()
                self.processed_events += 1

    def get_current_time(self) -> float:
        """Returns the current simulation time."""
        return self.current_time

    def setup_simulation(self, finish_time):
        """Sets stop condition and some params?!"""
        pass

    def stop(self):
        """Stops the simulation."""
        pass


class Request:
    def __init__(self, flow_id: int, arrival_time: float, resource_required: int):
        """
        Initialize a request.
        :param flow_id: ID of the flow generating the request
        :param arrival_time: Time at which the request arrives
        :param resource_required: Amount of network resource required for this request
        """
        self.flow_id = flow_id
        self.arrival_time = arrival_time

        # min_resources * service_time_with_min_resources
        self.total_size = size
        self.served_size = 0


        self.resource_required = resource_required
        self.start_service_time = None
        self.completion_time = None

    def set_service_start(self, start_time: float):
        """Records the time when the request starts service."""
        self.start_service_time = start_time

    def set_completion_time(self, completion_time: float):
        """Records the time when the request completes service."""
        self.completion_time = completion_time


class Flow:
    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        on_arrival_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize flow parameters and store statistics.
        :param flow_id: Unique identifier for the flow
        :param lambda_rate: Request generation rate (1 / average inter-arrival time)
        :param service_rate: Request service rate (1 / average service time)
        :param fixed_resource: Fixed resource allocation for requests (if any)
        :param dynamic_resource_range: Tuple (b_min, b_max) for dynamic resource allocation
        """
        self.flow_id = flow_id
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.on_arrival_callback = on_arrival_callback

        self.generated_requests = 0

    def set_on_arrival_callback(self, fn: Callable[[], None]):
        self.on_arrival_callback = fn

    def generate_request(self) -> Request:
        pass

    def generate_next_arrival_time(self) -> float:
        """Generates the time until the next request using exponential distribution."""
        return random.expovariate(self.arrival_rate)  # 1 / (1 / lambda)

    """TBD: logic of recalculation time for data traffic requests"""

    def generate_service_time(self, resources) -> float:
        """Generates the service time using exponential distribution."""
        return random.expovariate(self.service_rate)


class RealTimeFlow(Flow):
    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        fixed_resource: int,
    ):
        super().__init__(flow_id, arrival_rate, service_rate)
        self.fixed_resources = fixed_resource


class ElasticDataFlow(Flow):
    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        min_resources: int,
        max_resources: int,
    ):
        super().__init__(flow_id, arrival_rate, service_rate)
        self.min_resources = min_resources
        self.max_resources = max_resources


class Network:
    def __init__(self, capacity: int):
        """
        Initialize network parameters.
        :param capacity: Total network resource capacity (V)
        """
        self.capacity = capacity
        self.active_requests = []  # List to track ongoing requests

    def allocate_resources(self, request: Request):
        """
        Allocate network resources to a request.
        :param request: The request to allocate resources for
        """
        if request.resource_required <= self.capacity:
            self.active_requests.append(request)
            self.capacity -= request.resource_required
            return True
        return False

    def release_resources(self, request: Request):
        """
        Release resources held by a request when it completes service.
        """
        if request in self.active_requests:
            self.active_requests.remove(request)
            self.capacity += request.resource_required

    def redistribute_resources(self):
        """
        Dynamically adjust resource allocation among requests.
        """
        # Logic for redistribution goes here (if any)
        pass
