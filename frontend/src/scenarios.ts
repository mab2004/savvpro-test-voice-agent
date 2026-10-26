import { PresetScenario } from './types';

export const MESSY_CALLER_SCENARIOS: PresetScenario[] = [
  {
    id: 'mid-call-overwrite-confirm',
    title: '1. Change Mind Before Confirming',
    badge: 'Mid-Call Change',
    description:
      'Caller books for 4 people tomorrow at 7:00 PM, then right before confirming changes to 6 people at 8:30 PM.',
    turns: [
      'Hi, I would like to book an indoor table for 4 people tomorrow at 7:00 PM',
      'My name is Tariq Mahmood and my number is 0300-5551234',
      "Wait, my uncle's family is joining too, make it 6 people at 8:30 PM instead",
      'Yes, please confirm the booking',
    ],
  },
  {
    id: 'minglish-urdu-fractions',
    title: '2. Urdu / Minglish Conversation',
    badge: 'Urdu / Minglish',
    description:
      'Caller speaks in Pakistani Minglish ("kal saarhe saat baje char logon", "sawa aath baje", "haan jee bilkul confirm kar dein").',
    turns: [
      'Salam, kal saarhe saat baje char logon ka indoor table book karna hai',
      'Sorry, saarhe saat nahi, sawa aath baje kar do',
      'Mera naam Bilal Ahmed hai, number 0321-4455667',
      'Haan jee bilkul confirm kar dein',
    ],
  },
  {
    id: 'peak-hour-conflict',
    title: '3. 8:00 PM Full — Pick Another Time',
    badge: 'Busy Time Conflict',
    description:
      'Caller asks for 6 guests tomorrow at 8:00 PM Indoor (which is fully booked), hears alternatives, and picks 8:30 PM.',
    turns: [
      'Book an indoor table for 6 people tomorrow at 8:00 PM under Ayesha Siddiqui, phone 0333-9876543',
      "Let's do 8:30 PM instead",
      'Yes, confirm it please',
    ],
  },
  {
    id: 'faq-digression-stack',
    title: '4. Ask Questions Mid-Booking (Parking & Rain)',
    badge: 'Mid-Call Questions',
    description:
      'When asked for a phone number, the caller asks about valet parking and rooftop rain protection before giving their number.',
    turns: [
      'Hi, table for 4 people tomorrow at 7:30 PM on the rooftop under Kamran Ali',
      'Wait, do you guys have valet parking?',
      'And is the rooftop open if it rains?',
      'Awesome, my phone number is zero three zero zero double five four four three two one',
      'Yes, confirm please',
    ],
  },
  {
    id: 'homophones-self-correction',
    title: '5. Self-Correction in One Sentence',
    badge: 'Self-Correction',
    description:
      'Caller corrects themselves mid-sentence ("for 4 people—no wait, actually 5 people—tomorrow at 8 pm").',
    turns: [
      'Book a table for for people—no wait, actually 5 people—tomorrow at ate pm indoor',
      'Name is Nadia Hussain and phone is 0345-1122334',
      'Yes confirm',
    ],
  },
  {
    id: 'modify-or-cancel-seeded',
    title: '6. Change an Existing Booking (DD-1001)',
    badge: 'Modify Booking',
    description:
      'Caller quotes booking reference DD-1001 and moves their reservation time to 9:00 PM.',
    turns: [
      'I need to modify my existing reservation DD-1001',
      'Change the time to 9:00 PM please',
      'Yes, confirm the change',
    ],
  },
];

export function getContextualSuggestions(
  dialogueState: string,
  expectedSlot: string | null
): { label: string; text: string }[] {
  if (dialogueState === 'CONFIRMING_BOOKING') {
    return [
      {
        label: 'Yes, confirm the booking',
        text: 'Yes, please confirm the booking',
      },
      {
        label: 'Haan jee confirm kar dein',
        text: 'Haan jee bilkul confirm kar dein',
      },
      {
        label: 'Wait, change to 6 people at 8:30 PM',
        text: "Wait, my uncle's family is joining too, make it 6 people at 8:30 PM instead",
      },
      {
        label: 'Do you have valet parking?',
        text: 'Wait, do you guys have valet parking?',
      },
    ];
  }

  if (dialogueState === 'RESOLVING_CONFLICT') {
    return [
      {
        label: "Let's do 7:30 PM instead",
        text: "Let's do 7:30 PM instead",
      },
      {
        label: "Let's do 8:30 PM instead",
        text: "Let's do 8:30 PM instead",
      },
      {
        label: 'Switch to Rooftop at 8:00 PM',
        text: 'Switch to Rooftop please',
      },
    ];
  }

  if (dialogueState === 'CLARIFYING_AMBIGUITY') {
    return [
      {
        label: 'I meant 6 guests',
        text: '6 people',
      },
      {
        label: 'I meant 6:00 PM',
        text: '6:00 PM',
      },
    ];
  }

  if (expectedSlot === 'customer_name' || expectedSlot === 'phone_number') {
    return [
      {
        label: 'Tariq Mahmood, 0300-5551234',
        text: 'My name is Tariq Mahmood and my phone number is 0300-5551234',
      },
      {
        label: 'Bilal Ahmed, 0321-4455667 (Urdu)',
        text: 'Mera naam Bilal Ahmed hai, number 0321-4455667',
      },
      {
        label: 'Ask: Do you have valet parking?',
        text: 'Wait, do you guys have valet parking?',
      },
      {
        label: 'Change party size to 5',
        text: 'Actually, make it 5 people instead',
      },
    ];
  }

  if (expectedSlot === 'time') {
    return [
      {
        label: '7:30 PM',
        text: '7:30 PM',
      },
      {
        label: '8:00 PM',
        text: '8:00 PM',
      },
      {
        label: 'Saarhe saat baje (7:30 PM)',
        text: 'Saarhe saat baje',
      },
    ];
  }

  if (expectedSlot === 'date') {
    return [
      {
        label: 'Tomorrow',
        text: 'Tomorrow',
      },
      {
        label: 'Today',
        text: 'Today',
      },
      {
        label: 'Kal (Tomorrow)',
        text: 'Kal',
      },
    ];
  }

  return [
    {
      label: 'Table for 4 tomorrow at 7:30 PM',
      text: 'Hi, I would like to book an indoor table for 4 people tomorrow at 7:30 PM',
    },
    {
      label: '3 people—sorry, 5 people—at 8 PM',
      text: 'Book a table for 3 people—sorry, 5 people—tomorrow at 8 PM indoor',
    },
    {
      label: 'Kal 7:30 baje 4 logon ka table',
      text: 'Salam, kal saarhe saat baje char logon ka indoor table book karna hai',
    },
    {
      label: 'Do you have valet parking?',
      text: 'Do you guys have valet parking at F-7 Markaz?',
    },
  ];
}
