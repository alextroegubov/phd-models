from utils import ParametersSet
import itertools
import logging
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Callable, List
import random
from enum import Enum

from typing import Optional

from abc import abstractmethod, ABC

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
    # singleton class
    CANCELED_EVENT_TIME = -100
    _instance = None
    num_instances = 0

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            Simulator._instance = Simulator()
        return Simulator._instance

    def __init__(self):

        assert (
            self.__class__.num_instances == 0
        ), "Simulator class already exists somewhere"

        self.current_time = 0.0
        self.event_id_counter = 0
        self.event_queue = PriorityQueue()
        self.event_lookup: dict[int, Event] = {}

        self.processed_events = 0

        self.__class__.num_instances += 1

    def _get_next_id(self):
        new_id = self.event_id_counter
        self.event_id_counter += 1
        return new_id

    def schedule_event(
        self, time: float, action: Callable[[], None], event_type: EventType
    ) -> int:
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

    def stop(self):
        """Stops the simulation."""
        pass


class Request:
    def __init__(self, flow_id: int, arrival_time: float, total_size: float):
        """
        Initialize a request.

        :param flow_id: ID of the flow generating the request
        :param arrival_time: Time at which the request arrives
        :param resource_required: Amount of network resource required for this request
        """
        # flow id
        self.flow_id = flow_id
        # time the request is generated and sent to network
        self.arrival_time = arrival_time
        # min_resources * service_time_with_min_resources, total request_size
        self.total_size = total_size

        # served request size
        self.served_size = 0
        # last time current_alloc_resource changed
        self.last_alloc_time = self.arrival_time

        self.current_alloc_resource = 0
        self.total_alloc_resources = 0

        self.end_service_time = 0

    def _update_served_size_and_last_alloc_time(self):
        time = Simulator.get_instance().get_current_time()
        served_size = (time - self.arrival_time) * self.current_alloc_resource
        assert served_size >= 0, f"negative {served_size}"

        self.served_size += served_size
        self.last_alloc_time = time

    def realloc_resources(self, new_alloc_resources) -> float:
        self._update_served_size_and_last_alloc_time()
        self.current_alloc_resources = new_alloc_resources
        left_for_service = self.total_size - self.served_size
        service_time_left = left_for_service / self.current_alloc_resources

        assert service_time_left >= service_time_left
        return service_time_left


class Flow(ABC):
    def __init__(
        self,
        flow_id: int,
        arrival_rate: float,
        service_rate: float,
        on_arrival_callback: Optional[Callable[[Request], None]] = None,
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

    def set_on_arrival_callback(self, fn: Callable[[Request], None]):
        self.on_arrival_callback = fn

    def generate_next_arrival_time(self) -> float:
        """Generates the time until the next request using exponential distribution."""
        return random.expovariate(self.arrival_rate)  # 1 / (1 / lambda)

    def generate_request(self):
        size = self.generate_service_time() * self._get_min_resources()
        request = Request(
            flow_id=self.flow_id,
            arrival_time=Simulator.get_instance().get_current_time(),
            total_size=size,
        )

        self.on_arrival_callback(request)

        Simulator.get_instance().schedule_event(
            time=self.generate_next_arrival_time() + Simulator.get_instance().get_current_time(),
            action=self.generate_request,
            event_type=EventType.REQUEST_ARRIVAL,
        )

    @abstractmethod
    def _get_min_resources(self) -> int:
        pass

    @abstractmethod
    def _generate_service_time(self) -> float:
        pass


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

    def _get_min_resources(self) -> int:
        return self.fixed_resource

    def _generate_service_time(self) -> float:
        """Generates the service time using exponential distribution."""
        return random.expovariate(self.service_rate)


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

    def _get_min_resources(self) -> int:
        return self.min_resources

    def _generate_service_time(self) -> float:
        pass
        """Generates the service time using exponential distribution."""
        return random.expovariate(self.service_rate)


class Network:
    def __init__(self, capacity: int):
        """
        Initialize network parameters.
        :param capacity: Total network resource capacity (V)
        """
        self.capacity = capacity
        self.requests_lookup = {}
        self.flows_lookup = {}

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
